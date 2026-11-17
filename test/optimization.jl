using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems
import OptimizationVariableInterface

# Eliminable landmark (for the Schur solvers); needs create_ba_problem-style helpers from optimizeba.jl
struct OptLandmark{T}
    p::SVector{3, T}
end
OptimizationVariableInterface.nvars(::OptLandmark) = static(3)
OptimizationVariableInterface.update(v::OptLandmark, step, offset=1) = OptLandmark(v.p + SVector(step[offset], step[offset+1], step[offset+2]))
OptimizationVariableInterface.iseliminable(::Type{<:OptLandmark}) = static(true)
NLLSModel.generatemeasurement(camera, pt::OptLandmark) = NLLSModel.generatemeasurement(camera, pt.p)

function optlandmark_problem(ncams, nlms)
    problem = Model(Union{SVector{6, Float64}, OptLandmark{Float64}}, SimpleError2{2, Float64, SVector{6, Float64}, OptLandmark{Float64}})
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i in 1:ncams
        addvariable!(problem, randn(SVector{6, Float64}) .+ camoffset)
    end
    for i in 1:nlms
        addvariable!(problem, OptLandmark{Float64}(rand(SVector{3, Float64}) .+ SVector(-0.5, -0.5, 10.0)))
    end
    for c in 1:ncams, l in 1:nlms
        if (c + l) % 2 == 0 || c == 1
            addcost!(problem, SimpleError2{SVector{6, Float64}, OptLandmark{Float64}}(NLLSModel.generatemeasurement(problem.variables[c]::SVector{6, Float64}, problem.variables[ncams + l]), c, ncams + l))
        end
    end
    # Perturb the landmarks
    for i in ncams+1:ncams+nlms
        problem.variables[i] = OptLandmark{Float64}(problem.variables[i].p + 0.01 * randn(SVector{3, Float64}))
    end
    return problem
end

# Allocation of one call, after a warm-up call (in a function, so the measurement sees concrete types)
warmallocs_fdf(fdf!, g, H, x) = (fdf!(g, H, x); @allocated fdf!(g, H, x))
warmallocs_f(f, x) = (f(x); @allocated f(x))

function solve(problem; iterator=:levenbergmarquardt, reduction=:auto, storage=:auto, maxiters=100)
    c0 = NLLSModel.cost(problem)
    res = optimize!(problem, OptimizationOptions(iterator=iterator, maxiters=maxiters); numthreads=static(1), reduction, storage)
    return c0, NLLSModel.cost(problem), res
end

@testset "optimization.jl" begin
    # Every linear system and iterator takes steps that reduce the cost of a noise-free bundle adjustment problem
    Random.seed!(1)
    @testset "dense and sparse, $iterator" for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :newton)
        for (ncams, nlms) in ((3, 5), (10, 60))
            problem = perturb_ba_problem(create_ba_problem(ncams, nlms, 0.5), 0.01, 0.0)
            c0, c1, _ = solve(problem; iterator=iterator)
            @test c1 < 1e-3 * c0
        end
    end

    @testset "Schur, $storage, $iterator" for storage in (:dense, :sparse), iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd)
        problem = optlandmark_problem(4, 20)
        c0, c1, _ = solve(problem; iterator=iterator, reduction=:eliminate, storage=storage)
        @test c1 < 1e-3 * c0
    end

    # Dense and sparse Schur reach the same solution
    Random.seed!(4)
    p1 = optlandmark_problem(4, 20)
    p2 = deepcopy(p1)
    _, c_dense, _ = solve(p1; reduction=:eliminate, storage=:dense)
    _, c_sparse, _ = solve(p2; reduction=:eliminate, storage=:sparse)
    @test c_dense ≈ c_sparse atol=1e-8

    # Threaded Schur linearization (disjoint landmark ownership, shared C and E) matches the serial result
    @testset "threaded Schur, $storage" for storage in (:dense, :sparse)
        Random.seed!(5)
        p_serial = optlandmark_problem(30, 120)
        p_threads = deepcopy(p_serial)
        @test NLLSModel.workestimates(p_serial, nothing).normal >= 2 * NLLSModel.MIN_WORK_PER_THREAD_NORMAL
        f1, fdf1!, x1, H1 = optimvars(p_serial; numthreads=static(1), reduction=:eliminate, storage)
        f4, fdf4!, x4, H4 = optimvars(p_threads; numthreads=static(4), reduction=:eliminate, storage)
        @test fdf4!.linsystem isa NLLSModel.LinearSystemShared
        @test fdf4!.plan isa NLLSModel.SchurThreadPlan
        n = H1.structure.nr + sum(H1.structure.e_dims)
        g1 = zeros(n); g4 = zeros(n)
        for _ in 1:2 # Evaluate twice: buffers must be re-zeroed
            @test fdf1!(g1, H1, x1) ≈ fdf4!(g4, H4, x4)
            @test g1 ≈ g4
            dx = randn(n)
            @test dot(dx, H1, dx) ≈ dot(dx, H4, dx)
        end
        @test f1(x1) ≈ f4(x4)
        # Floors for exactly 2 of 8 threads: 2 buffers, and the same results as one thread
        w = NLLSModel.workestimates(p_serial, nothing)
        f2, fdf2!, x2, H2 = optimvars(deepcopy(p_serial); numthreads=static(8), reduction=:eliminate, storage, min_work=w.normal ÷ 2, min_work_cost=w.cost ÷ 2)
        @test length(fdf2!.linsystem.ls) == 2 && f2.nthreads == 2
        # At most MAX_THREADS_NORMAL buffers, however many threads
        fdf8! = optimvars(deepcopy(p_serial); numthreads=static(8), reduction=:eliminate, storage, min_work=0)[2]
        @test length(fdf8!.linsystem.ls) == NLLSModel.MAX_THREADS_NORMAL
        g2 = zeros(n)
        @test fdf1!(g1, H1, x1) ≈ fdf2!(g2, H2, x2) rtol=1e-13
        @test g1 ≈ g2 rtol=1e-12
        @test f1(x1) ≈ f2(x2) rtol=1e-13
        # Threaded, a warm call allocates nothing: the linearization, and the cost (threaded too, above its floor). The
        # linearization takes the threaded path (four buffers) even on one Julia thread, so rules.md 21's 1.10 exception
        # applies to it.
        @test f4.nthreads > 1
        @test threaded_allocs_ok(warmallocs_fdf(fdf4!, g4, H4, x4), length(fdf4!.linsystem.ls) > 1)
        @test warmallocs_f(f4, x4) == 0
    end
end

@testset "Schur pattern is handed to the linear system" begin
    Random.seed!(6)
    prob = optlandmark_problem(4, 20)
    _, _, _, H = optimvars(prob; numthreads=static(1), reduction=:eliminate, storage=:sparse)
    @test H.pattern !== nothing
    @test size(H.pattern.S_sparse, 1) == H.structure.nr
    _, _, _, Hd = optimvars(prob; numthreads=static(1), reduction=:eliminate, storage=:dense)
    @test Hd.pattern === nothing
    # The solver accepts it and optimizes
    c0, c1, _ = solve(optlandmark_problem(4, 20); reduction=:eliminate, storage=:sparse)
    @test c1 < 1e-3 * c0
end
