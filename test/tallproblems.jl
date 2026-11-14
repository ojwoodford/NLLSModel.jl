using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
import OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars

# Test problems shared by tallkernels.jl and tallformulation.jl. Needs Tgt from frontdoor.jl.

# --- Test problems ---------------------------------------------------------------------------------------------------

# Exponential decay a exp(-b t) + c fitted to one sample, on variable `i` (a Vector{Float64} of length 3)
struct TlExp <: NLLSModel.AbstractResidual
    t::Float64
    y::Float64
    i::Int
end
NLLSModel.ndeps(::TlExp) = static(1)
NLLSModel.nres(::TlExp) = static(1)
NLLSModel.varindices(r::TlExp) = r.i
NLLSModel.getvars(r::TlExp, vars::AbstractVector) = (vars[r.i]::Vector{Float64},)
NLLSModel.computeresidual(r::TlExp, x) = SVector(x[1] * exp(-x[2] * r.t) + x[3] - r.y)
Base.eltype(::TlExp) = Float64

# A residual with a robust kernel
struct TlRob{R <: NLLSModel.AbstractResidual, K <: AbstractRobustifier} <: NLLSModel.AbstractResidual
    res::R
    kernel::K
end
NLLSModel.ndeps(r::TlRob) = ndeps(r.res)
NLLSModel.nres(r::TlRob) = nres(r.res)
NLLSModel.varindices(r::TlRob) = varindices(r.res)
NLLSModel.getvars(r::TlRob, vars::AbstractVector) = getvars(r.res, vars)
NLLSModel.computeresidual(r::TlRob, args...) = computeresidual(r.res, args...)
NLLSModel.robustkernel(r::TlRob) = r.kernel
Base.eltype(r::TlRob) = eltype(r.res)

# A rotation, updated by right-multiplying with the exponential of the step (three dofs)
struct TlRot{T}
    R::SMatrix{3, 3, T, 9}
end
nvars(::TlRot) = static(3)
function tlexp(w)
    s = w[1]^2 + w[2]^2 + w[3]^2
    K = SMatrix{3, 3}(zero(s), w[3], -w[2], -w[3], zero(s), w[1], w[2], -w[1], zero(s))
    if s < 1e-8
        A = 1 - s / 6
        B = 0.5 - s / 24
    else
        θ = sqrt(s)
        A = sin(θ) / θ
        B = (1 - cos(θ)) / s
    end
    return one(K) + A * K + B * (K * K)
end
function update(v::TlRot, step, offset=1)
    return TlRot(v.R * tlexp(SVector(step[offset], step[offset + 1], step[offset + 2])))
end
Base.:(==)(a::TlRot, b::TlRot) = a.R == b.R
Base.isapprox(a::TlRot, b::TlRot; kwargs...) = isapprox(a.R, b.R; kwargs...)

# Rotating a point should give the target, for the rotation variable `i`
struct TlPt <: NLLSModel.AbstractResidual
    p::SVector{3, Float64}
    q::SVector{3, Float64}
    i::Int
end
NLLSModel.ndeps(::TlPt) = static(1)
NLLSModel.nres(::TlPt) = static(3)
NLLSModel.varindices(r::TlPt) = r.i
NLLSModel.getvars(r::TlPt, vars::AbstractVector) = (vars[r.i]::TlRot{Float64},)
NLLSModel.computeresidual(r::TlPt, v) = v.R * r.p - r.q
Base.eltype(::TlPt) = Float64

# A residual that is not a residual
struct TlCost <: NLLSModel.AbstractCost
    i::Int
end
NLLSModel.ndeps(::TlCost) = static(1)
NLLSModel.varindices(c::TlCost) = c.i
NLLSModel.getvars(c::TlCost, vars::AbstractVector) = (vars[c.i]::Vector{Float64},)
NLLSModel.computecost(::TlCost, x) = sum(abs2, x)
Base.eltype(::TlCost) = Float64

# An adaptive residual, whose first variable is the kernel
struct TlAdapt <: AbstractAdaptiveResidual
    varind::SVector{2, Int}
end
NLLSModel.ndeps(::TlAdapt) = static(2)
NLLSModel.nres(::TlAdapt) = static(1)
NLLSModel.varindices(r::TlAdapt) = r.varind
NLLSModel.getvars(r::TlAdapt, vars::Vector) = (vars[r.varind[1]]::ContaminatedGaussian{Float64}, vars[r.varind[2]]::Float64)
NLLSModel.computeresidual(::TlAdapt, x) = SVector(x - 1.0)
Base.eltype(::TlAdapt) = Float64

tlkernel(h, w=0.05) = h === :second ? Huber2oKernel(w) : HuberKernel(w) # `huber`: false, true (first order) or :second

# Noisy exponential decay with a few outliers, with a Huber kernel on every cost if `huber`
function curvefit(; n=40, huber=false, outliers=0, seed=1, width=0.05)
    rng = MersenneTwister(seed)
    truth = [2.0, 1.3, 0.5]
    ts = range(0, 3; length=n)
    ys = [truth[1] * exp(-truth[2] * t) + truth[3] + 0.01 * randn(rng) for t in ts]
    ys[1:outliers] .+= 3.0
    cost(t, y) = huber !== false ? TlRob(TlExp(t, y, 1), tlkernel(huber, width)) : TlExp(t, y, 1)
    c0 = cost(ts[1], ys[1])
    model = Model(Vector{Float64}, typeof(c0))
    addvariable!(model, truth .+ [0.3, -0.2, 0.2])
    for (t, y) in zip(ts, ys)
        addcost!(model, cost(t, y))
    end
    return model
end

# A rotation fitted to noisy points
function rotproblem(seed=2)
    rng = MersenneTwister(seed)
    Rtrue = tlexp(SVector(0.4, -0.7, 0.2))
    model = Model(TlRot{Float64}, TlPt)
    addvariable!(model, TlRot(Rtrue * tlexp(0.3 * randn(rng, SVector{3, Float64}))))
    for _ in 1:12
        p = randn(rng, SVector{3, Float64})
        addcost!(model, TlPt(p, Rtrue * p + 0.01 * randn(rng, SVector{3, Float64}), 1))
    end
    return model
end

# Variables of different sizes and update rules: a Vector{Float64} of 3, a rotation, an SVector{2} and a Vector{Float64} of 1
function posemodel(seed=3; huber=false, width=0.05)
    rng = MersenneTwister(seed)
    Rtrue = tlexp(SVector(-0.3, 0.5, 0.9))
    truth = [2.0, 1.3, 0.5]
    t2 = SVector(0.7, -1.1)
    t4 = [0.4]
    model = Model(Union{Vector{Float64}, TlRot{Float64}, SVector{2, Float64}}, Union{TlExp, TlRob{TlExp, typeof(HuberKernel(0.05))}, TlRob{TlExp, typeof(Huber2oKernel(0.05))}, TlPt, Tgt{SVector{2, Float64}}, Tgt{Vector{Float64}}})
    addvariable!(model, truth .+ [0.3, -0.2, 0.2])
    addvariable!(model, TlRot(Rtrue * tlexp(0.3 * randn(rng, SVector{3, Float64}))))
    addvariable!(model, t2 + 0.3 * randn(rng, SVector{2, Float64}))
    addvariable!(model, t4 .+ 0.3 .* randn(rng, 1))
    for t in range(0, 3; length=15)
        c = TlExp(t, truth[1] * exp(-truth[2] * t) + truth[3] + 0.01 * randn(rng), 1)
        addcost!(model, huber !== false ? TlRob(c, tlkernel(huber, width)) : c)
    end
    for _ in 1:6
        p = randn(rng, SVector{3, Float64})
        addcost!(model, TlPt(p, Rtrue * p + 0.01 * randn(rng, SVector{3, Float64}), 2))
    end
    addcost!(model, Tgt(t2, 3))
    addcost!(model, Tgt(t4, 4))
    return model
end

# optimvars_nls with every variable unfixed returns x0 *as* model.variables, and the evaluators copy each point they evaluate into
# the model, so evaluating anywhere else overwrites x0. optimize! takes a copy; so do the tests that evaluate by hand.
tlnls(model; kwargs...) = optimvars_nls(model; kwargs...)
costlist(model) = collect(Iterators.flatten(values(model.costs)))
samevars(a, b; atol=1e-5) = all(isapprox(u, v; atol=atol) for (u, v) in zip(a.variables, b.variables))

