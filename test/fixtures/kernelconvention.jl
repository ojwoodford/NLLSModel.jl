using NLLSModel, Test, Static, StaticArrays, Random
using OptimizationVariableInterface: update

fixture("allocations")

# Composite Simpson's rule for ∫₀ᴿ f(x) dx with `panels` (even) panels
function simpson(f, R, panels)
    h = R / panels
    acc = f(zero(R)) + f(R)
    for i in 1:panels-1
        acc += (isodd(i) ? 4 : 2) * f(i * h)
    end
    return acc * h / 3
end

# The measure of the unit sphere in ℝᵈ, d = 1, 2, 3
const SPHERE_MEASURE = (2.0, 2π, 4π)

# ∫ p(r) dr over ℝᵈ, by radial quadrature of p = exp(−ρ(‖r‖²)/2) to R = 40 times the larger standard deviation. `minus` is
# subtracted from p before integrating (the uniform term of GaussianPlusUniform, which is not integrable over ℝᵈ).
function densityintegral(kernel, d, sigmamax; minus=0.0, panels=400_000)
    return SPHERE_MEASURE[d] * simpson(x -> (exp(-robustify(kernel, x^2) / 2) - minus) * x^(d - 1), 40 * sigmamax, panels)
end

# A ContaminatedGaussian for residuals of length d
cgkernel(s1, s2, w, d) = ContaminatedGaussian(s1, s2, w, static(d))

# `n` seeded residuals of length d, a fraction `w` from N(0, σ₁²I) and the rest from N(0, σ₂²I); their squared norms
function mixturesqnorms(rng, n, d, s1, s2, w)
    return [sum(abs2, (rand(rng) < w ? s1 : s2) .* randn(rng, d)) for _ in 1:n]
end

# Central differences, in BigFloat so that rounding plays no part (the kernels are generic in the type of `s`)
fdiff1(f, x, h) = (f(x + h) - f(x - h)) / (2h)
fdiff2(f, x, h) = (f(x + h) - 2 * f(x) + f(x - h)) / h^2

# Squared norms at which the kernels' derivatives are checked (Huber's kink, at the width 1 used, is not one of them)
const KERNEL_FD_COSTS = (0.01, 0.09, 0.49, 1.69, 4.0, 25.0)

# The first and second derivative of robustify(kernel, s) in s, by central differences in BigFloat
fdcost1(kernel, s) = fdiff1(Base.Fix1(robustify, kernel), big(s), big(1e-20) * max(s, 1))
fdcost2(kernel, s) = fdiff2(Base.Fix1(robustify, kernel), big(s), big(1e-20) * max(s, 1))

# The gradient and Hessian of robustify(update(kernel, x), s + x[end]) in x at 0, by central differences in BigFloat
function fdkernel(kernel, s)
    n = Int(dynamic(NLLSModel.nvars(kernel))) + 1
    h = big(1e-20)
    f(x) = robustify(update(kernel, x), s + x[end])
    e(i) = [j == i ? h : zero(h) for j in 1:n]
    g = [(f(e(i)) - f(-e(i))) / 2h for i in 1:n]
    H = [(f(e(i) + e(j)) - f(e(i) - e(j)) - f(-e(i) + e(j)) + f(-e(i) - e(j))) / (4h^2) for i in 1:n, j in 1:n]
    return g, H
end

# An adaptive residual of length 2, r = x − data, with the kernel (of type K) as variable 1
struct PairAdaptive{K} <: NLLSModel.AbstractAdaptiveResidual
    data::SVector{2, Float64}
    varind::Int
end
PairAdaptive{K}(data) where K = PairAdaptive{K}(data, 2)
NLLSModel.ndeps(::PairAdaptive) = static(2)
NLLSModel.nres(::PairAdaptive) = static(2)
NLLSModel.varindices(r::PairAdaptive) = SVector(1, r.varind)
NLLSModel.getvars(r::PairAdaptive{K}, vars::Vector) where K = (vars[1]::K, vars[r.varind]::SVector{2, Float64})
NLLSModel.computeresidual(r::PairAdaptive, x) = x - r.data
Base.eltype(::PairAdaptive) = Float64

# A residual of length 2 with a fixed kernel of its own
struct PairFixed{K} <: NLLSModel.AbstractResidual
    data::SVector{2, Float64}
    kernel::K
end
NLLSModel.ndeps(::PairFixed) = static(1)
NLLSModel.nres(::PairFixed) = static(2)
NLLSModel.varindices(::PairFixed) = 1
NLLSModel.getvars(::PairFixed, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(r::PairFixed, x) = x - r.data
NLLSModel.robustkernel(r::PairFixed) = r.kernel
Base.eltype(::PairFixed) = Float64

# A user's kernel, which defines no kerneldim
struct UserKernel <: AbstractRobustifier end
NLLSModel.robustify(::UserKernel, s) = s
NLLSModel.robustifydcost(::UserKernel, s) = (s, one(s), zero(s))

# The cost, gradient and Hessian of `computecost(res, update(kernel, x), vars + x)` at x = 0 by central differences, with
# the kernel's parameters first (`withkernel = false`: the variable's only)
function fdcostgradhess(res, kernel, var; withkernel=true, h=1e-5)
    nk = withkernel ? Int(dynamic(NLLSModel.nvars(kernel))) : 0
    nv = length(var)
    n = nk + nv
    f(x) = NLLSModel.computecost(res, withkernel ? update(kernel, x, 1) : kernel, var + SVector{nv}(ntuple(i -> x[nk + i], nv)))
    e(i) = SVector{n, Float64}(ntuple(j -> j == i ? h : 0.0, n))
    z = zeros(SVector{n, Float64})
    g = [(f(e(i)) - f(-e(i))) / 2h for i in 1:n]
    H = [(f(e(i) + e(j)) - f(e(i) - e(j)) - f(-e(i) + e(j)) + f(-e(i) - e(j))) / 4h^2 for i in 1:n, j in 1:n]
    return f(z), g, H
end

# A GaussianPlusUniform for residuals of length d, with outlier volume A
gpukernel(sigma, w, d; volume=50.0) = GaussianPlusUniform(sigma, w, static(d); volume)

# An adaptive residual of length 2 on a run-time sized variable (a `Vector`), with a hand-written Jacobian in buffers it
# owns, so that the Hessian path takes its run-time size (`Matrix` Jacobian) branch
struct PairAdaptiveDynamic{K} <: NLLSModel.AbstractAdaptiveResidual
    data::SVector{2, Float64}
    jac::Matrix{Float64}
end
PairAdaptiveDynamic{K}(data) where K = PairAdaptiveDynamic{K}(data, zeros(2, 3))
NLLSModel.ndeps(::PairAdaptiveDynamic) = static(2)
NLLSModel.nres(::PairAdaptiveDynamic) = static(2)
NLLSModel.varindices(::PairAdaptiveDynamic) = SVector(1, 2)
NLLSModel.getvars(::PairAdaptiveDynamic{K}, vars::Vector) where K = (vars[1]::K, vars[2]::Vector{Float64})
NLLSModel.computeresidual(r::PairAdaptiveDynamic, x) = SVector(x[1] * x[2], x[3]) - r.data
function NLLSModel.computeresjac(varflags, r::PairAdaptiveDynamic, x)
    r.jac[1, 1] = x[2]; r.jac[1, 2] = x[1]; r.jac[1, 3] = 0.0
    r.jac[2, 1] = 0.0;  r.jac[2, 2] = 0.0;  r.jac[2, 3] = 1.0
    return NLLSModel.computeresidual(r, x), r.jac
end
Base.eltype(::PairAdaptiveDynamic) = Float64

# A residual of length 2 with its own outlier volume, handed to a fixed GaussianPlusUniform through robustkernel
struct PairVolume{K} <: NLLSModel.AbstractResidual
    data::SVector{2, Float64}
    logvolume::Float64
    kernel::K
end
NLLSModel.ndeps(::PairVolume) = static(1)
NLLSModel.nres(::PairVolume) = static(2)
NLLSModel.varindices(::PairVolume) = 1
NLLSModel.getvars(::PairVolume, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(r::PairVolume, x) = x - r.data
NLLSModel.robustkernel(r::PairVolume) = withlogvolume(r.kernel, r.logvolume)
Base.eltype(::PairVolume) = Float64

# The evaluations whose allocations are measured (in a function, so that the measurement sees concrete types)
costgradhess(varflags, res, kernel, var) = NLLSModel.computecostgradhess(varflags, res, kernel, var)

# An adaptive residual of length N, r = x − data, x an SVector{N}
struct VecAdaptive{N, K} <: NLLSModel.AbstractAdaptiveResidual
    data::SVector{N, Float64}
end
NLLSModel.ndeps(::VecAdaptive) = static(2)
NLLSModel.nres(::VecAdaptive{N}) where N = static(N)
NLLSModel.varindices(::VecAdaptive) = SVector(1, 2)
NLLSModel.getvars(::VecAdaptive{N, K}, vars::Vector) where {N, K} = (vars[1]::K, vars[2]::SVector{N, Float64})
NLLSModel.computeresidual(r::VecAdaptive, x) = x - r.data
Base.eltype(::VecAdaptive) = Float64
