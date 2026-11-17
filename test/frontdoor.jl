using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems
import OptimizationVariableInterface

# The front door: optimize!(model, options) and optimize(model, options) (the extension), and the `reduction` and `storage`
# keywords of optimvars. Needs optlandmark_problem from optimization.jl.

include("fixtures/frontdoor.jl")

@testset "frontdoor.jl" begin
    @testset "variables hold the solution: $name" for (name, V, n) in (("static 1", SVector{1, Float64}, 1), ("static 16", SVector{16, Float64}, 16), ("static 17", SVector{17, Float64}, 17), ("dynamic Vector", Vector{Float64}, 17), ("short Vector", Vector{Float64}, 3))
        model, t = singleproblem(V, n, 1)
        c0 = NLLSModel.cost(model)
        result = optimize!(model, FDOptions)
        @test result isa OptimizationResult
        @test isconverged(result)
        @test model.variables[1] ≈ t atol=1e-6
        @test NLLSModel.cost(model) ≈ result.bestcost atol=1e-12
        @test result.startcost ≈ c0
    end

    @testset "variables hold the solution: several variables, $name" for (name, n) in (("dense", 5), ("sparse", 60))
        model, targets = chainproblem(n, 2, 2)
        _, _, _, H = optimvars(model; numthreads=static(1))
        @test isnothing(H) == (name == "dense") # Check the storage the test is meant to exercise
        result = optimize!(model, FDOptions; numthreads=static(1))
        @test issolved(model, targets)
        @test NLLSModel.cost(model) ≈ result.bestcost atol=1e-12
    end

    @testset "variables hold the solution: Schur, $storage" for storage in (:dense, :sparse)
        Random.seed!(3)
        model = optlandmark_problem(4, 20)
        _, _, _, H = optimvars(model; numthreads=static(1), storage)
        @test H isa BlockLinearSystems.SchurBlockHessian
        @test (H.R isa Matrix) == (storage === :dense)
        c0 = NLLSModel.cost(model)
        result = optimize!(model, FDOptions; storage)
        @test NLLSModel.cost(model) ≈ result.bestcost atol=1e-12
        @test NLLSModel.cost(model) < 1e-3 * c0
    end

    @testset "write-back order, unfixed = $name" for (name, unfixed, free) in (
            ("all", nothing, 1:7),
            ("an index", 6, [6]),
            ("indices", [1, 3, 4, 7], [1, 3, 4, 7]),
            ("a type", Vector{Float64}, [1, 3, 4, 6, 7]),
            ("a mask", Bool[1, 0, 1, 1, 0, 0, 1], [1, 3, 4, 7]))
        model, targets = mixedproblem()
        start = deepcopy(model.variables)
        if length(free) > 1 # block_sizes_indices has to re-sort these
            mask = NLLSModel.convertunfixed(unfixed, model)
            @test !NLLSModel.block_sizes_indices(model.variables, mask)[4]
        end
        result = optimize!(model, FDOptions; unfixed, numthreads=static(1))
        @test issolved(model, targets, free)
        for k in setdiff(1:7, free)
            @test model.variables[k] == start[k] # Fixed variables are untouched, bit for bit
        end
        # Fixed variables only start away from their targets
        @test !issolved(model, targets, setdiff(1:7, free)) || length(free) == 7
    end

    @testset "optimize leaves the model alone" begin
        model, targets = chainproblem(5, 2, 4)
        start = deepcopy(model.variables)
        c0 = NLLSModel.cost(model)
        solved, result = optimize(model, FDOptions)
        @test model.variables == start
        @test NLLSModel.cost(model) == c0
        @test solved !== model
        @test issolved(solved, targets)
        @test NLLSModel.cost(solved) ≈ result.bestcost atol=1e-12
        # A single static variable too
        model, t = singleproblem(SVector{4, Float64}, 4, 5)
        start = model.variables[1]
        solved, _ = optimize(model, FDOptions)
        @test model.variables[1] == start
        @test solved.variables[1] ≈ t atol=1e-6
    end

    @testset "same as the manual route, cost for cost: $name" for (name, make) in (
            ("single static", () -> singleproblem(SVector{3, Float64}, 3, 6)[1]),
            ("single dynamic", () -> singleproblem(Vector{Float64}, 20, 6)[1]),
            ("dense", () -> chainproblem(5, 2, 6)[1]),
            ("sparse", () -> chainproblem(60, 2, 6)[1]),
            ("Schur", () -> (Random.seed!(6); optlandmark_problem(4, 20))))
        m1 = make()
        m2 = deepcopy(m1)
        traj1, traj2 = CostTrajectory(), CostTrajectory()
        r1 = optimize!(m1, FDOptions; numthreads=static(1), callback=storecostscallback(traj1))
        f, fdf!, x0, H = optimvars(m2; numthreads=static(1))
        obj = HessianObjective(f, fdf!; H)
        if OptimizationVariableInterface.ismutablevar(x0)
            r2 = optimize!(x0, obj, FDOptions; callback=storecostscallback(traj2))
            syncvars!(m2, fdf!, x0)
        else
            x, r2 = optimize(x0, obj, FDOptions; callback=storecostscallback(traj2))
            syncvars!(m2, fdf!, x)
        end
        @test traj1.costs == traj2.costs
        @test !isempty(traj1.costs)
        @test r1.startcost == r2.startcost && r1.bestcost == r2.bestcost && r1.niterations == r2.niterations
        @test m1.variables == m2.variables
    end

    @testset "workspace and callback are passed on" begin
        model, targets = chainproblem(5, 2, 7)
        calls = Ref(0)
        counter(cost, unused...) = (calls[] += 1; (cost, 0))
        optimize!(model, FDOptions; callback=counter, workspace=OptimizationWorkspace(), numthreads=static(1))
        @test calls[] > 0
        @test issolved(model, targets)
    end

    # AN5: a workspace with elimination on the normal path. Each call builds a new Schur structure, so whether a second
    # call reuses the workspace is not tested (see the fixes file, "To decide").
    @testset "a workspace with elimination, storage = :$storage" for storage in (:dense, :sparse)
        Random.seed!(11)
        ba = optlandmark_problem(4, 20)
        solved(; kwargs...) = (model = deepcopy(ba); optimize!(model, FDOptions; reduction=:eliminate, storage, numthreads=static(1), kwargs...); NLLSModel.cost(model))
        ref = solved()
        @test ref < NLLSModel.cost(ba)
        workspace = OptimizationWorkspace()
        @test solved(; workspace) ≈ ref rtol=1e-10
        @test solved(; workspace) ≈ ref rtol=1e-10 # The same workspace again
    end

    # This package's static rule for one variable (`usesstaticsystem`, `MAX_STATIC_SYSTEM`), on its own: SecondOrderOptimization
    # has a twin, deliberately not shared, and the two may differ
    @testset "static rule for one variable: $name" for (name, V, n, isstatic) in (("1", SVector{1, Float64}, 1, true),
            ("16", SVector{16, Float64}, 16, true), ("17", SVector{17, Float64}, 17, false), ("dynamic 3", Vector{Float64}, 3, false),
            ("dynamic 17", Vector{Float64}, 17, false))
        model, _ = singleproblem(V, n, 8)
        _, fdf!, x0, H = optimvars(model; numthreads=static(1))
        @test NLLSModel.MAX_STATIC_SYSTEM == 16
        @test NLLSModel.usesstaticsystem(x0) == isstatic
        @test (typeof(fdf!.linsystem.ls) <: NLLSModel.UniVariateLSstatic_ls) == isstatic
    end

    # For one variable, the H that optimvars returns has the types of the linear system's own storage, so buffers built
    # from it (H by deepcopy, g by `zero(H[:, 1])`) take fdf!'s direct-write branch: the costs accumulate straight into
    # them, with no internal accumulation and no copy. Other buffer types take the fallback, with the same result.
    @testset "single-variable prototype: $name" for (name, V, n, nt) in (
            ("static 3", SVector{3, Float64}, 3, 1), ("static 16", SVector{16, Float64}, 16, 1),
            ("static 3, shared", SVector{3, Float64}, 3, 4), ("static 17", SVector{17, Float64}, 17, 1),
            ("dynamic 3", Vector{Float64}, 3, 1), ("dynamic 3, shared", Vector{Float64}, 3, 4))
        model = linsingleproblem(V, n, 11)
        _, fdf!, x0, H = optimvars(model; numthreads=static(nt), min_work=0) # min_work=0: share the system when nt > 1
        @test (fdf!.linsystem isa NLLSModel.LinearSystemShared) == (nt > 1)
        isstatic = V <: StaticVector && n <= 16
        @test H isa (isstatic ? MMatrix{n, n, Float64, n * n} : Matrix{Float64})
        @test size(H) == (n, n)
        @test all(iszero, H)
        Hd = deepcopy(H)
        gd = zero(H[:, 1])
        @test gd isa (isstatic ? MVector{n, Float64} : Vector{Float64})
        internal = NLLSModel.getA(fdf!.linsystem)
        fill!(internal, 7.0)
        c = fdf!(gd, Hd, x0)
        @test all(==(7.0), internal) # The direct branch never touches the internal storage
        @test singlefdf_allocs(fdf!, gd, Hd, x0) == 0 # Threaded too, when the suite runs on several threads
        # The fallback: buffers of another type than the system's
        gf, Hf = isstatic ? (zeros(n), zeros(n, n)) : (view(zeros(n), :), view(zeros(n, n), :, :))
        cf = fdf!(gf, Hf, x0)
        @test !all(==(7.0), internal) # The fallback accumulates internally, then copies
        @test cf == c
        @test gf == gd
        @test Hf == Hd
    end

    @testset "reduction and storage" begin
        Random.seed!(9)
        ba = optlandmark_problem(4, 20) # Has eliminable variables
        plain, _ = chainproblem(5, 2, 9) # Has none
        issch(model; kwargs...) = optimvars(model; numthreads=static(1), kwargs...)[4] isa BlockLinearSystems.SchurBlockHessian

        @testset "each reduction value" begin
            @test issch(ba) && issch(ba; reduction=:auto) && issch(ba; reduction=:eliminate)
            @test !issch(ba; reduction=:none)
            @test !issch(plain) && !issch(plain; reduction=:auto) && !issch(plain; reduction=:none)
            @test_throws ArgumentError optimvars(plain; numthreads=static(1), reduction=:eliminate)
            @test_throws ArgumentError optimvars(ba; numthreads=static(1), reduction=:project)
            @test_throws ArgumentError optimvars(ba; numthreads=static(1), reduction=:schur)
            @test_throws ArgumentError optimvars(ba; numthreads=static(1), storage=:both)
            @test_throws ArgumentError optimvars(plain; unfixed=1, numthreads=static(1), reduction=:project) # Even for a single variable
            # Each one solves, and elimination gives the same optimum as none
            costs = map((:auto, :none, :eliminate)) do reduction
                model = deepcopy(ba)
                optimize!(model, FDOptions; reduction)
                NLLSModel.cost(model)
            end
            @test all(c -> c < 1e-6, costs)
        end

        @testset "storage of the reduced system" begin
            _, _, _, Hd = optimvars(ba; numthreads=static(1), storage=:dense)
            _, _, _, Hs = optimvars(ba; numthreads=static(1), storage=:sparse)
            @test Hd.R isa Matrix{Float64}
            @test Hs.R isa BlockLinearSystems.SparseBlockLinearSystem
            md, ms = deepcopy(ba), deepcopy(ba)
            optimize!(md, FDOptions; storage=:dense)
            optimize!(ms, FDOptions; storage=:sparse)
            @test NLLSModel.cost(md) ≈ NLLSModel.cost(ms) atol=1e-8
            @test all(k -> isapprox(landmarkvec(md.variables[k]), landmarkvec(ms.variables[k]); atol=1e-5), eachindex(md.variables))
        end

        # With nothing eliminated, storage chooses the Hessian's storage: :auto keeps the structure's choice
        @testset "storage without elimination" begin
            small, _ = chainproblem(5, 2, 9)  # 10 dimensions: :auto chooses dense
            large, _ = chainproblem(60, 2, 9) # 120 dimensions, a chain: :auto chooses sparse
            Hof(model; kw...) = optimvars(model; numthreads=static(1), kw...)[4]
            @test Hof(small) === nothing && Hof(large) isa SparseMatrixCSC # :auto, unchanged
            @test Hof(small; storage=:sparse) isa SparseMatrixCSC
            @test Hof(large; storage=:dense) === nothing
            @test Hof(ba; reduction=:none, storage=:sparse) isa SparseMatrixCSC && Hof(ba; reduction=:none, storage=:dense) === nothing
            # The same solution either way
            for (model, forced) in ((small, :sparse), (large, :dense))
                a, b = deepcopy(model), deepcopy(model)
                optimize!(a, FDOptions; numthreads=static(1))
                optimize!(b, FDOptions; numthreads=static(1), storage=forced)
                @test NLLSModel.cost(a) ≈ NLLSModel.cost(b) atol=1e-10
                @test all(k -> isapprox(a.variables[k], b.variables[k]; atol=1e-6), eachindex(a.variables))
            end
            # A single variable's Hessian is always dense
            single, _ = singleproblem(Vector{Float64}, 5, 1)
            err = try optimvars(single; storage=:sparse); nothing catch e e end
            @test err isa ArgumentError && occursin("single", err.msg)
            @test optimvars(single; storage=:dense)[4] isa Matrix{Float64}
        end

        # max_dense_bytes: :auto stores a dense matrix above the cap sparse instead, :dense throws there
        @testset "max_dense_bytes on the normal path" begin
            small, _ = chainproblem(5, 2, 9) # A dense 10×10 Hessian is 800 bytes
            Hof(model; kw...) = optimvars(model; numthreads=static(1), kw...)[4]
            @test Hof(small; max_dense_bytes=800) === nothing
            @test Hof(small; max_dense_bytes=799) isa SparseMatrixCSC
            @test Hof(small; storage=:dense, max_dense_bytes=800) === nothing
            err = try Hof(small; storage=:dense, max_dense_bytes=799); nothing catch e e end
            @test err isa ArgumentError && occursin("10×10", err.msg) && occursin("800", err.msg)
            # The reduced Hessian under elimination: 4 cameras of 6, so 24×24, 4608 bytes
            @test Hof(ba; storage=:dense, max_dense_bytes=4608).R isa Matrix{Float64}
            @test_throws ArgumentError Hof(ba; storage=:dense, max_dense_bytes=4607)
            @test Hof(ba; storage=:auto, max_dense_bytes=4607).R isa BlockLinearSystems.SparseBlockLinearSystem
            # A single variable has no sparse alternative: above the cap it throws either way
            single, _ = singleproblem(Vector{Float64}, 5, 1) # 200 bytes
            @test Hof(single; max_dense_bytes=200) isa Matrix{Float64}
            @test_throws ArgumentError Hof(single; max_dense_bytes=199)
            @test_throws ArgumentError Hof(single; storage=:dense, max_dense_bytes=199)
            # And through the front door
            m = deepcopy(small)
            optimize!(m, FDOptions; numthreads=static(1), max_dense_bytes=799)
            @test NLLSModel.cost(m) ≈ NLLSModel.cost(optimize(small, FDOptions; numthreads=static(1))[1]) atol=1e-10
            @test_throws ArgumentError optimize!(deepcopy(small), FDOptions; storage=:dense, max_dense_bytes=799)
        end

        @testset "the linsolver keyword is gone" begin
            @test_throws MethodError optimvars(ba; numthreads=static(1), linsolver=:schur)
            @test_throws MethodError optimvars(plain; numthreads=static(1), linsolver=:cholesky)
            @test_throws MethodError optimize!(deepcopy(plain), FDOptions; linsolver=:schur)
        end
    end
end
