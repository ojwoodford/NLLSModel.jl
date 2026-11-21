using NLLSModel, Test, Static, StaticArrays, Random

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
