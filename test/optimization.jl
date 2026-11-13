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

function solve(problem; iterator=:levenbergmarquardt, linsolver=nothing, maxiters=100)
    f, fdf!, x0, H = optimvars(problem, nothing, static(1); linsolver=linsolver)
    c0 = f(x0)
    options = OptimizationOptions(iterator=iterator, maxiters=maxiters)
    x_opt, res = optimize(f, fdf!, x0, options; H=H)
    NLLSModel.syncvars!(problem, fdf!, x_opt)
    return c0, NLLSModel.cost(problem), res
end

@testset "optimization.jl" begin
    # Every linear system and iterator takes steps that reduce the cost of a noise-free bundle adjustment problem
    Random.seed!(3)
    @testset "dense and sparse, $iterator" for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :newton)
        for (ncams, nlms) in ((3, 5), (10, 60))
            problem = perturb_ba_problem(create_ba_problem(ncams, nlms, 0.5), 0.01, 0.0)
            c0, c1, _ = solve(problem; iterator=iterator)
            @test c1 < 1e-3 * c0
        end
    end

    @testset "Schur, $linsolver, $iterator" for linsolver in (:dense_schur, :sparse_schur), iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd)
        problem = optlandmark_problem(4, 20)
        c0, c1, _ = solve(problem; iterator=iterator, linsolver=linsolver)
        @test c1 < 1e-3 * c0
    end

    # Dense and sparse Schur reach the same solution
    Random.seed!(4)
    p1 = optlandmark_problem(4, 20)
    p2 = deepcopy(p1)
    _, c_dense, _ = solve(p1; linsolver=:dense_schur)
    _, c_sparse, _ = solve(p2; linsolver=:sparse_schur)
    @test c_dense ≈ c_sparse atol=1e-8

    # Threaded Schur linearization (disjoint landmark ownership, shared C and E) matches the serial result
    @testset "threaded Schur, $linsolver" for linsolver in (:dense_schur, :sparse_schur)
        Random.seed!(5)
        p_serial = optlandmark_problem(30, 120)
        p_threads = deepcopy(p_serial)
        @test NLLSModel.countcosts(NLLSModel.costwork, p_serial.costs) >= NLLSModel.MIN_WORK_FOR_THREADED_LINEARIZATION
        f1, fdf1!, x1, H1 = optimvars(p_serial, nothing, static(1); linsolver=linsolver)
        f4, fdf4!, x4, H4 = optimvars(p_threads, nothing, static(4); linsolver=linsolver)
        @test fdf4!.linsystem isa NLLSModel.LinearSystemShared
        @test fdf4!.plan isa NLLSModel.SchurThreadPlan
        n = H1.structure.nr + sum(H1.structure.e_dims)
        g1 = zeros(n); g4 = zeros(n)
        for _ in 1:2 # Evaluate twice: buffers must be re-zeroed
            @test fdf1!(g1, H1, x1) ≈ fdf4!(g4, H4, x4)
            @test g1 ≈ g4
            dx = randn(n)
            @test BlockLinearSystems.fast_bAb(H1, dx) ≈ BlockLinearSystems.fast_bAb(H4, dx)
        end
        @test f1(x1) ≈ f4(x4)
    end
end

@testset "Schur pattern is handed to the linear system" begin
    Random.seed!(6)
    prob = optlandmark_problem(4, 20)
    _, _, _, H = optimvars(prob, nothing, static(1); linsolver=:sparse_schur)
    @test H.pattern !== nothing
    @test size(H.pattern.S_sparse, 1) == H.structure.nr
    _, _, _, Hd = optimvars(prob, nothing, static(1); linsolver=:dense_schur)
    @test Hd.pattern === nothing
    # The solver accepts it and optimizes
    c0, c1, _ = solve(optlandmark_problem(4, 20); linsolver=:sparse_schur)
    @test c1 < 1e-3 * c0
end
