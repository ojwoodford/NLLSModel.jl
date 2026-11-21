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
