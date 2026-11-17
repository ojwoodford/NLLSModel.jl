using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra, Random
import SecondOrderOptimization
import SecondOrderOptimization: optimize!, OptimizationOptions, HessianObjective
import OptimizationVariableInterface, BlockLinearSystems

# Variables of any size. A variable that becomes a block of a block linear system is limited only by BlockLinearSystems,
# which checks block sizes itself (this package states no limit); a variable optimized on its own, or on the plain
# residual-and-Jacobian path, has none.

# Pins a variable of any size to a measurement
struct PinRes <: NLLSModel.AbstractResidual
    meas::Vector{Float64}
    i::Int
end
NLLSModel.ndeps(::PinRes) = static(1)
NLLSModel.nres(r::PinRes) = length(r.meas)
NLLSModel.varindices(r::PinRes) = r.i
NLLSModel.getvars(r::PinRes, vars::AbstractVector) = (vars[r.i]::Vector{Float64},)
NLLSModel.computeresidual(r::PinRes, x) = x .- r.meas
Base.eltype(::PinRes) = Float64

# Links the first three entries of one variable to another variable of three dimensions
struct LinkRes <: NLLSModel.AbstractResidual
    i::Int
    j::Int
end
NLLSModel.ndeps(::LinkRes) = static(2)
NLLSModel.nres(::LinkRes) = static(3)
NLLSModel.varindices(r::LinkRes) = SVector(r.i, r.j)
NLLSModel.getvars(r::LinkRes, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::Vector{Float64})
NLLSModel.computeresidual(::LinkRes, a, b) = SVector(a[1] - b[1], a[2] - b[2], a[3] - b[3])
Base.eltype(::LinkRes) = Float64

# Eliminable three-dimensional variable, for the Schur systems
struct SmallLandmark{T}
    p::SVector{3, T}
end
OptimizationVariableInterface.nvars(::SmallLandmark) = static(3)
OptimizationVariableInterface.update(v::SmallLandmark, step, offset=1) = SmallLandmark(v.p + SVector(step[offset], step[offset+1], step[offset+2]))
OptimizationVariableInterface.iseliminable(::Type{<:SmallLandmark}) = static(true)

struct LandmarkRes <: NLLSModel.AbstractResidual
    i::Int
    j::Int
end
NLLSModel.ndeps(::LandmarkRes) = static(2)
NLLSModel.nres(::LandmarkRes) = static(3)
NLLSModel.varindices(r::LandmarkRes) = SVector(r.i, r.j)
NLLSModel.getvars(r::LandmarkRes, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::SmallLandmark{Float64})
NLLSModel.computeresidual(::LandmarkRes, a, l) = SVector(a[1] - l.p[1], a[2] - l.p[2], a[3] - l.p[3])
Base.eltype(::LandmarkRes) = Float64

function schurproblem(n)
    problem = Model(Union{Vector{Float64}, SmallLandmark{Float64}}, LandmarkRes)
    addvariable!(problem, zeros(n))
    for _ in 1:3
        addvariable!(problem, SmallLandmark(SVector(1.0, 2.0, 3.0)))
    end
    for j in 2:4
        addcost!(problem, LandmarkRes(1, j))
    end
    return problem
end

const CostUnion = Union{PinRes, LinkRes}

# An eliminable variable of any (run-time) size
struct BigLandmark{T}
    p::Vector{T}
end
OptimizationVariableInterface.nvars(v::BigLandmark) = length(v.p)
OptimizationVariableInterface.update(v::BigLandmark, step, offset=1) = BigLandmark(v.p + view(step, offset:offset+length(v.p)-1))
OptimizationVariableInterface.iseliminable(::Type{<:BigLandmark}) = static(true)

# A retained variable of three dimensions seeing a BigLandmark: as many rows as the landmark has dimensions
struct BigSight <: NLLSModel.AbstractResidual
    w::Matrix{Float64}
    z::Vector{Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::BigSight) = static(2)
NLLSModel.nres(r::BigSight) = length(r.z)
NLLSModel.varindices(r::BigSight) = SVector(r.i, r.j)
NLLSModel.getvars(r::BigSight, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::BigLandmark{Float64})
NLLSModel.computeresidual(r::BigSight, p, l) = l.p .* (1 .+ 0.1 .* sin(p[3])) .- r.w * p .- r.z
Base.eltype(::BigSight) = Float64

# Four retained variables of three dimensions (each pinned), and `nlms` eliminable ones of `n`, each seen by two of them
function bigelimproblem(n; nlms=3, seed=7)
    rng = MersenneTwister(seed)
    problem = Model(Union{Vector{Float64}, BigLandmark{Float64}}, Union{PinRes, BigSight})
    poses = [randn(rng, 3) for _ in 1:4]
    foreach(p -> addvariable!(problem, p .+ 0.1 .* randn(rng, 3)), poses)
    foreach(((k, p),) -> addcost!(problem, PinRes(p, k)), enumerate(poses))
    for j in 1:nlms
        l = randn(rng, n)
        lj = addvariable!(problem, BigLandmark(l .+ 0.1 .* randn(rng, n)))
        for k in (mod1(j, 4), mod1(j + 1, 4))
            w = randn(rng, n, 3)
            addcost!(problem, BigSight(w, l .* (1 .+ 0.1 .* sin(poses[k][3])) .- w * poses[k] .+ 0.01 .* randn(rng, n), k, lj))
        end
    end
    return problem
end

# Variable 1 has `n` dimensions; variables 2 to `nsmall + 1` have three
function bigproblem(n, nsmall=3)
    problem = Model(Vector{Float64}, CostUnion)
    addvariable!(problem, zeros(n))
    for _ in 1:nsmall
        addvariable!(problem, [1.0, 2.0, 3.0])
    end
    addcost!(problem, PinRes(collect(1.0:n), 1))
    for j in 2:nsmall+1
        addcost!(problem, PinRes([1.0, 2.0, 3.0], j))
        addcost!(problem, LinkRes(1, j))
    end
    return problem
end

@testset "bigvariable.jl" begin
    # A variable of any size can be added
    problem = bigproblem(256)
    @test length(problem.variables[1]) == 256

    # It can be optimized on its own, and the single variable system does not use blocks
    f, fdf!, x0, H = optimvars(problem; unfixed=1, numthreads=static(1))
    @test H isa Matrix{Float64} && size(H) == (256, 256)
    res = optimize!(x0, HessianObjective(f, fdf!; H), OptimizationOptions(iterator=:levenbergmarquardt, maxiters=20))
    x = x0
    @test f(x) < 1e-8
    @test x ≈ collect(1.0:256)

    # The residual-and-Jacobian path writes a flat Jacobian, so it accepts a larger variable too
    f, res!, resjac!, x0, J = optimvars_nls(problem; unfixed=1, numthreads=static(1))
    r = zeros(size(J, 1))
    Jm = copy(J)
    @test resjac!(r, Jm, x0) ≈ f(x0)
    @test size(J, 2) == 256

    # A variable of 256 dimensions (past the old limit of 255) is a block: retained, in the plain system (dense and sparse) and
    # in a Schur system, and eliminated; every route reaches the same cost. The variables are Vectors, so any size past the
    # limit runs the same code
    opts = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=100)
    solved(problem; kwargs...) = NLLSModel.cost(first(SecondOrderOptimization.optimize(problem, opts; numthreads=static(1), kwargs...)))
    @testset "a retained block of $n dimensions" for n in (256,)
        problem = bigproblem(n)
        problem.costs[PinRes][2] = PinRes([1.5, 2.0, 3.0], 2) # In conflict with the link to variable 1, so the minimum is not zero
        c0 = NLLSModel.cost(problem)
        cn = solved(problem; reduction=:none, storage=:dense)
        @test 0 < cn < 0.1 * c0
        @test optimvars(problem; numthreads=static(1), storage=:sparse)[4] isa SparseMatrixCSC
        @test solved(problem; reduction=:none, storage=:sparse) ≈ cn rtol=1e-8
        @test solved(problem; formulation=:tall, storage=:dense) ≈ cn rtol=1e-8
        @test solved(problem; formulation=:tall, storage=:sparse) ≈ cn rtol=1e-8
        # Retained in a Schur system
        schur = schurproblem(n)
        @test optimvars(schur; numthreads=static(1))[4] isa BlockLinearSystems.SchurBlockHessian
        for storage in (:dense, :sparse)
            @test solved(schur; reduction=:eliminate, storage) ≈ solved(schur; reduction=:none) atol=1e-12
            @test solved(schur; formulation=:tall, reduction=:eliminate, storage) ≈ solved(schur; reduction=:none) atol=1e-12
        end
    end
    @testset "eliminated blocks of $n dimensions" for n in (256,)
        problem = bigelimproblem(n)
        c0 = NLLSModel.cost(problem)
        cn = solved(problem; reduction=:none)
        @test cn < 0.01 * c0
        @test optimvars(problem; numthreads=static(1))[4] isa BlockLinearSystems.SchurBlockHessian
        for storage in (:dense, :sparse)
            @test solved(problem; reduction=:eliminate, storage) ≈ cn rtol=1e-8
            @test solved(problem; formulation=:tall, reduction=:eliminate, storage) ≈ cn rtol=1e-8
        end
    end
end
