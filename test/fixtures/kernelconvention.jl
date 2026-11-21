using NLLSModel, Test, Static, StaticArrays, Random
using OptimizationVariableInterface: update

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

# A ContaminatedGaussian for residuals of length d. Today's constructor takes no d, so it is ignored.
cgkernel(s1, s2, w, d) = ContaminatedGaussian(s1, s2, w)

# The kernel fitted to squared residual norms `s`. Today's fit.
fitkernel(kernel, s; maxiters=100) = NLLSModel.optimizekernel(kernel, s, maxiters)

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
