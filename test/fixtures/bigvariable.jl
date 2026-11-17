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
