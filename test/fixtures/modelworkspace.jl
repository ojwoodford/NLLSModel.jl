using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization

fixture("frontdoor", "optimization")

# Links two static vectors of equal length: a - b = d
struct SLnk{N} <: NLLSModel.AbstractResidual
    d::SVector{N, Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::SLnk) = static(2)
NLLSModel.nres(::SLnk{N}) where N = static(N)
NLLSModel.varindices(r::SLnk) = SVector(r.i, r.j)
NLLSModel.getvars(r::SLnk{N}, vars::AbstractVector) where N = (vars[r.i]::SVector{N, Float64}, vars[r.j]::SVector{N, Float64})
NLLSModel.computeresidual(r::SLnk, a, b) = a - b - r.d
Base.eltype(::SLnk) = Float64

# A cost that is not a residual: a quartic bowl with its minimum at `t`
struct QuartCost <: NLLSModel.AbstractCost
    t::SVector{2, Float64}
    i::Int
end
NLLSModel.ndeps(::QuartCost) = static(1)
NLLSModel.varindices(c::QuartCost) = c.i
NLLSModel.getvars(c::QuartCost, vars::AbstractVector) = (vars[c.i]::SVector{2, Float64},)
NLLSModel.computecost(c::QuartCost, x) = sum(abs2, x - c.t) + sum(abs2, abs2.(x - c.t))
Base.eltype(::QuartCost) = Float64

# A chain of `n` static 2-vectors, each pinned to its target and linked to its neighbour (allocation-free to linearize)
function staticchain(n; seed=1)
    rng = MersenneTwister(seed)
    targets = [randn(rng, SVector{2, Float64}) for _ in 1:n]
    model = Model(SVector{2, Float64}, Union{Tgt{SVector{2, Float64}}, SLnk{2}})
    foreach(k -> addvariable!(model, targets[k] + 0.3 * randn(rng, SVector{2, Float64})), 1:n)
    foreach(k -> addcost!(model, Tgt(targets[k], k)), 1:n)
    foreach(k -> addcost!(model, SLnk(targets[k] - targets[k+1], k, k + 1)), 1:n-1)
    return model
end

# Two static 2-vectors under costs that are not residuals
function quartmodel()
    model = Model(SVector{2, Float64}, QuartCost)
    addvariable!(model, SVector(1.0, 2.0))
    addvariable!(model, SVector(-1.0, 0.5))
    addcost!(model, QuartCost(SVector(0.3, 0.1), 1))
    addcost!(model, QuartCost(SVector(-0.2, 0.4), 2))
    return model
end

const MWOptions = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=50)

# A solve with a workspace, and the bytes a repeated one allocates from the same start (in a function, so the measurement
# sees concrete types)
function mw_allocs(model, opts, ws; kw...)
    start = copy(model.variables)
    optimize!(model, opts; kw..., workspace=ws)
    copyto!(model.variables, start)
    return @allocated optimize!(model, opts; kw..., workspace=ws)
end
