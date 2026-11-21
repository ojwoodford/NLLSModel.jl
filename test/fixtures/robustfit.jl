using NLLSModel, Test, Static, StaticArrays, Random

fixture("allocations", "kernelconvention")

# `n` seeded squared norms of residuals of length d: a fraction `w` from N(0, σ²I), the rest uniform over the cube of side
# `side` centred at 0 (volume side^d)
function uniformsqnorms(rng, n, d, sigma, w, side)
    return [sum(abs2, rand(rng) < w ? sigma .* randn(rng, d) : side .* (rand(rng, d) .- 0.5)) for _ in 1:n]
end

# The fit's objective: the model's cost Σ 0.5 ρ(sᵢ) plus the prior's term −(α − 1) log w − (β − 1) log(1 − w)
function fitobjective(kernel, s, logvolume=nothing; wprior=(1.0, 1.0))
    α, β = wprior
    w = NLLSModel.params(kernel)[end]
    c = isnothing(logvolume) ? sum(si -> 0.5 * robustify(kernel, si), s) :
                               sum(0.5 * robustify(withlogvolume(kernel, l), si) for (si, l) in zip(s, logvolume))
    return c - (α - 1) * log(w) - (β - 1) * log(1 - w)
end

# An adaptive residual of length 2, r = x − data, that hands the kernel over with its own log volume
struct PairVolumeAdaptive{K} <: NLLSModel.AbstractAdaptiveResidual
    data::SVector{2, Float64}
    logvolume::Float64
    varind::SVector{2, Int}
end
NLLSModel.ndeps(::PairVolumeAdaptive) = static(2)
NLLSModel.nres(::PairVolumeAdaptive) = static(2)
NLLSModel.varindices(r::PairVolumeAdaptive) = r.varind
NLLSModel.getvars(r::PairVolumeAdaptive{K}, vars::Vector) where K =
    (withlogvolume(vars[r.varind[1]]::K, r.logvolume), vars[r.varind[2]]::SVector{2, Float64})
NLLSModel.computeresidual(r::PairVolumeAdaptive, x) = x - r.data
Base.eltype(::PairVolumeAdaptive) = Float64

# The fits whose allocations are measured (in a function, so that the measurement sees concrete types)
fitlow(kernel, s) = optimizerobustifier(kernel, s)
fitlow(kernel, s, logvolume) = optimizerobustifier(kernel, s, logvolume)
fitmodel(model, index, s, logvolume) = optimizerobustifier!(model, index; s, logvolume)

import SecondOrderOptimization: optimize!, OptimizationOptions

# Alternation: `nvar` (1 or 2) unfixed 2-vectors, each the mean of 400 residuals, 80 % from N(mean, I) and the rest outliers
# (N(mean, 10²I) for a ContaminatedGaussian, uniform over a square of side 40 for a GaussianPlusUniform), and the kernel,
# variable 1, which the residuals hand over as it is. Returns the model and the true means.
function alternationmodel(kernel; nvar=2, nper=400, seed=11, start=nothing)
    K = typeof(kernel)
    rng = MersenneTwister(seed)
    truths = (SVector(1.0, -1.0), SVector(-2.0, 0.5))[1:nvar]
    model = Model(Union{K, SVector{2, Float64}}, PairAdaptive{K})
    addvariable!(model, kernel)
    for m in truths
        addvariable!(model, isnothing(start) ? SVector(0.0, 0.0) : m + start)
    end
    for (i, m) in enumerate(truths), _ in 1:nper
        y = rand(rng) < 0.8 ? m + randn(rng, SVector{2, Float64}) :
            kernel isa GaussianPlusUniform ? 40 .* (rand(rng, SVector{2, Float64}) .- 0.5) : m + 10 .* randn(rng, SVector{2, Float64})
        addcost!(model, PairAdaptive{K}(y, i + 1))
    end
    return model, truths
end

# The kernel's true parameters in `alternationmodel`
alternationtruth(::ContaminatedGaussian) = SVector(1.0, 10.0, 0.8)
alternationtruth(::GaussianPlusUniform) = SVector(1.0, 0.8)

# Optimize `model` with the kernel (variable 1) fixed, refitting it in the callback. Returns the result, and per callback
# the cost the optimizer passed in, `cost(model)` before the fit, and the cost the callback returned.
function alternate!(model, options; kwargs...)
    s, logvolume = Float64[], Float64[]
    given, before, returned = Float64[], Float64[], Float64[]
    callback = (c, x, data, iteratedata) -> begin
        push!(given, c)
        push!(before, cost(model))
        r = optimizerobustifier!(model, 1; s, logvolume)
        push!(returned, r)
        (r, 0)
    end
    result = optimize!(model, options; callback, unfixed=2:length(model.variables), numthreads=static(1), kwargs...)
    return result, given, before, returned
end
