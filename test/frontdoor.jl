using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems
import OptimizationVariableInterface

# The front door: optimize!(model, options) and optimize(model, options) (the extension), and the `reduction` and `reduced`
# keywords of optimvars. Needs optlandmark_problem from optimization.jl.

# Pins a variable of any type to a target (nonlinear, with its minimum at the target)
struct Tgt{V} <: NLLSModel.AbstractResidual
    t::V
    i::Int
end
NLLSModel.ndeps(::Tgt) = static(1)
NLLSModel.nres(::Tgt{<:StaticVector{N}}) where N = static(N)
NLLSModel.nres(r::Tgt{<:Vector}) = length(r.t)
NLLSModel.varindices(r::Tgt) = r.i
NLLSModel.getvars(r::Tgt{V}, vars::AbstractVector) where V = (vars[r.i]::V,)
NLLSModel.computeresidual(r::Tgt, x) = (x .+ 0.1 .* x .^ 3) .- (r.t .+ 0.1 .* r.t .^ 3)
Base.eltype(::Tgt) = Float64

# Links two vectors of equal length: a - b = d
struct Lnk <: NLLSModel.AbstractResidual
    d::Vector{Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::Lnk) = static(2)
NLLSModel.nres(r::Lnk) = length(r.d)
NLLSModel.varindices(r::Lnk) = SVector(r.i, r.j)
NLLSModel.getvars(r::Lnk, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::Vector{Float64})
NLLSModel.computeresidual(r::Lnk, a, b) = a .- b .- r.d
Base.eltype(::Lnk) = Float64

const FDOptions = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=100)

# One variable of type `V`, started away from its target
function singleproblem(V, n, seed)
    rng = MersenneTwister(seed)
    t = V(randn(rng, n))
    model = Model(V, Tgt{V})
    addvariable!(model, V(t .+ 0.3 .* randn(rng, n)))
    addcost!(model, Tgt(t, 1))
    return model, t
end

# A chain of `n` vectors of dimension `d`, each pinned to its target and linked to its neighbour
function chainproblem(n, d, seed)
    rng = MersenneTwister(seed)
    targets = [randn(rng, d) for _ in 1:n]
    model = Model(Vector{Float64}, Union{Tgt{Vector{Float64}}, Lnk})
    for k in 1:n
        addvariable!(model, targets[k] .+ 0.3 .* randn(rng, d))
    end
    for k in 1:n
        addcost!(model, Tgt(targets[k], k))
    end
    for k in 1:n-1
        addcost!(model, Lnk(targets[k] .- targets[k+1], k, k+1))
    end
    return model, targets
end

# Vectors of 3, 1, 2 dimensions and static 2-vectors, in an order that block_sizes_indices has to re-sort. Pairs (1, 4) and
# (3, 7) are linked, and no linked pair is split by any `unfixed` below, so every unfixed variable has its target as optimum.
function mixedproblem()
    rng = MersenneTwister(11)
    sizes = [3, 2, 1, 3, 2, 2, 1]
    isstatic = [false, true, false, false, true, false, false]
    targets = Any[isstatic[k] ? SVector{2, Float64}(randn(rng, 2)) : randn(rng, sizes[k]) for k in 1:7]
    model = Model(Union{Vector{Float64}, SVector{2, Float64}}, Union{Tgt{Vector{Float64}}, Tgt{SVector{2, Float64}}, Lnk})
    for k in 1:7
        addvariable!(model, targets[k] .+ 0.3 .* (isstatic[k] ? SVector{2, Float64}(randn(rng, 2)) : randn(rng, sizes[k])))
    end
    for k in 1:7
        addcost!(model, Tgt(targets[k], k))
    end
    addcost!(model, Lnk(targets[1] .- targets[4], 1, 4))
    addcost!(model, Lnk(targets[3] .- targets[7], 3, 7))
    return model, targets
end

landmarkvec(v) = v isa OptLandmark ? v.p : v
issolved(model, targets, which=eachindex(targets); atol=1e-6) = all(k -> isapprox(model.variables[k], targets[k]; atol=atol), which)

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
        _, _, _, H = optimvars(model, nothing, static(1))
        @test isnothing(H) == (name == "dense") # Check the storage the test is meant to exercise
        result = optimize!(model, FDOptions; numthreads=static(1))
        @test issolved(model, targets)
        @test NLLSModel.cost(model) ≈ result.bestcost atol=1e-12
    end

    @testset "variables hold the solution: Schur, $reduced" for reduced in (:dense, :sparse)
        Random.seed!(3)
        model = optlandmark_problem(4, 20)
        _, _, _, H = optimvars(model, nothing, static(1); reduced)
        @test H isa BlockLinearSystems.SchurBlockMatrix
        @test (H.R isa Matrix) == (reduced === :dense)
        c0 = NLLSModel.cost(model)
        result = optimize!(model, FDOptions; reduced)
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
        f, fdf!, x0, H = optimvars(m2, nothing, static(1))
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

    @testset "static rule agrees with the optimizer: $name" for (name, V, n) in (("1", SVector{1, Float64}, 1), ("16", SVector{16, Float64}, 16), ("17", SVector{17, Float64}, 17), ("dynamic 3", Vector{Float64}, 3), ("dynamic 17", Vector{Float64}, 17))
        model, _ = singleproblem(V, n, 8)
        _, fdf!, x0, H = optimvars(model, nothing, static(1))
        ours = typeof(fdf!.linsystem.ls) <: NLLSModel.UniVariateLSstatic_ls
        theirs = makelinearsystem(x0, H, :auto) isa SecondOrderOptimization.StaticLinearSystem
        @test ours == theirs
        @test ours == usesstaticsystem(x0) # SecondOrderOptimization's one statement of the rule (MAX_STATIC_SYSTEM = 16)
        @test ours == (V <: StaticVector && n <= 16)
    end

    @testset "reduction and reduced" begin
        Random.seed!(9)
        ba = optlandmark_problem(4, 20) # Has eliminable variables
        plain, _ = chainproblem(5, 2, 9) # Has none
        issch(model; kwargs...) = optimvars(model, nothing, static(1); kwargs...)[4] isa BlockLinearSystems.SchurBlockMatrix

        @testset "each reduction value" begin
            @test issch(ba) && issch(ba; reduction=:auto) && issch(ba; reduction=:eliminate)
            @test !issch(ba; reduction=:none)
            @test !issch(plain) && !issch(plain; reduction=:auto) && !issch(plain; reduction=:none)
            @test_throws ArgumentError optimvars(plain, nothing, static(1); reduction=:eliminate)
            @test_throws ArgumentError optimvars(ba, nothing, static(1); reduction=:project)
            @test_throws ArgumentError optimvars(ba, nothing, static(1); reduction=:schur)
            @test_throws ArgumentError optimvars(ba, nothing, static(1); reduced=:both)
            @test_throws ArgumentError optimvars(plain, 1, static(1); reduction=:project) # Even for a single variable
            # Each one solves, and elimination gives the same optimum as none
            costs = map((:auto, :none, :eliminate)) do reduction
                model = deepcopy(ba)
                optimize!(model, FDOptions; reduction)
                NLLSModel.cost(model)
            end
            @test all(c -> c < 1e-6, costs)
        end

        @testset "reduced storage" begin
            _, _, _, Hd = optimvars(ba, nothing, static(1); reduced=:dense)
            _, _, _, Hs = optimvars(ba, nothing, static(1); reduced=:sparse)
            @test Hd.R isa Matrix{Float64}
            @test Hs.R isa BlockLinearSystems.SparseBlockLinearSystem
            md, ms = deepcopy(ba), deepcopy(ba)
            optimize!(md, FDOptions; reduced=:dense)
            optimize!(ms, FDOptions; reduced=:sparse)
            @test NLLSModel.cost(md) ≈ NLLSModel.cost(ms) atol=1e-8
            @test all(k -> isapprox(landmarkvec(md.variables[k]), landmarkvec(ms.variables[k]); atol=1e-5), eachindex(md.variables))
        end

        @testset "reduced is silently ignored without elimination, $reduced" for reduced in (:dense, :sparse)
            for (model, reduction) in ((ba, :none), (plain, :auto), (plain, :none))
                @test_nowarn optimvars(model, nothing, static(1); reduction, reduced)
                _, _, _, Hr = optimvars(model, nothing, static(1); reduction, reduced)
                _, _, _, Ha = optimvars(model, nothing, static(1); reduction)
                @test typeof(Hr) == typeof(Ha)
                trajr, traja = CostTrajectory(), CostTrajectory()
                mr, ma = deepcopy(model), deepcopy(model)
                @test_nowarn optimize!(mr, FDOptions; reduction, reduced, numthreads=static(1), callback=storecostscallback(trajr))
                optimize!(ma, FDOptions; reduction, numthreads=static(1), callback=storecostscallback(traja))
                @test trajr.costs == traja.costs
                @test mr.variables == ma.variables
            end
            # A single variable has nothing to eliminate either
            single, _ = singleproblem(Vector{Float64}, 5, 1)
            @test_nowarn optimvars(single, nothing, static(1); reduced)
        end

        @testset "the linsolver keyword is gone" begin
            @test_throws MethodError optimvars(ba, nothing, static(1); linsolver=:schur)
            @test_throws MethodError optimvars(plain, nothing, static(1); linsolver=:cholesky)
            @test_throws MethodError optimize!(deepcopy(plain), FDOptions; linsolver=:schur)
        end
    end
end
