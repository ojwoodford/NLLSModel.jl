# Adaptive robust kernels. Each returns `ρ(s) = −2 log p(r)` exactly, for a density `p` over `r ∈ ℝᴰ` with `s = ‖r‖²`,
# including every term that depends on its parameters or on `D`; the residual's cost is `0.5·ρ(s)`.

# The residual length `D` of an adaptive kernel, given as a `StaticInt` (so that it is a type parameter)
kerneldimension(::StaticInt{D}) where D = D >= 1 ? D : throw(ArgumentError("The residual length d of an adaptive kernel must be at least 1 (got static($D))"))
kerneldimension(d::Integer) = throw(ArgumentError("Pass the residual length d of an adaptive kernel as a StaticInt, static($d), not as $(typeof(d))"))

# `(D/2)·log 2π`, the normalization of a `D`-dimensional Gaussian
halflog2pi(::Type{T}, D) where T = T(D) * T(0.9189385332046727417803297364056176398613974736377834128171515404827656959)

# The two-term logaddexp of both kernels: for `a` and `b`, the logarithms of the two terms of the density, `ρ = −2 log(eᵃ + eᵇ)`
# and `γ`, the first term's responsibility, with one `exp` and one `log1p`, and no overflow for finite `a` and `b`
@inline function rhoresponsibility(a, b)
    t = a - b
    pos = t >= 0
    e = exp(pos ? -t : t)
    rho = -2 * ((pos ? a : b) + log1p(e))
    gamma = pos ? inv(1 + e) : e / (1 + e)
    return rho, gamma
end

struct ContaminatedGaussian{T<:Real, D} <: AbstractAdaptiveRobustifier
    invsigma1::ZeroToInfScalar{T}   # u₁ = invsigma1.logval = log(1/σ₁)
    invsigma2::ZeroToInfScalar{T}   # u₂
    w::ZeroToOneScalar{T}           # the first component's weight
    halfinvvar1::T                  # exp(2u₁)/2
    halfinvvar2::T                  # exp(2u₂)/2
    k1::T                           # log(w) + D·u₁ − (D/2)·log 2π
    k2::T                           # log(1 − w) + D·u₂ − (D/2)·log 2π
end
function ContaminatedGaussian(invsigma1::ZeroToInfScalar{T}, invsigma2::ZeroToInfScalar{T}, w::ZeroToOneScalar{T}, d::StaticInt) where T <: Real
    D = kerneldimension(d)
    u1 = invsigma1.logval
    u2 = invsigma2.logval
    L = halflog2pi(T, D)
    return ContaminatedGaussian{T, D}(invsigma1, invsigma2, w, exp(2 * u1) / 2, exp(2 * u2) / 2, log(w.val) + D * u1 - L, log(w.comp) + D * u2 - L)
end
ContaminatedGaussian(invsigma1::ZeroToInfScalar, invsigma2::ZeroToInfScalar, w::ZeroToOneScalar, d::Integer) = kerneldimension(d)
function ContaminatedGaussian(s1::Real, s2::Real, w::Real, d::Union{StaticInt, Integer})
    T = float(promote_type(typeof(s1), typeof(s2), typeof(w)))
    kerneldimension(d)
    return ContaminatedGaussian(ZeroToInfScalar{T}(inv(T(s1))), ZeroToInfScalar{T}(inv(T(s2))), ZeroToOneScalar{T}(T(w)), d)
end
nvars(::ContaminatedGaussian) = static(3)
update(var::ContaminatedGaussian{T, D}, updatevec, start=1) where {T, D} = ContaminatedGaussian(update(var.invsigma1, updatevec, start), update(var.invsigma2, updatevec, start+1), update(var.w, updatevec, start+2), static(D))
params(var::ContaminatedGaussian) = SVector(1.0 / var.invsigma1.val, 1.0 / var.invsigma2.val, var.w.val)
kerneldim(::ContaminatedGaussian{T, D}) where {T, D} = D

@inline cgterms(kernel::ContaminatedGaussian, cost) = (kernel.k1 - cost * kernel.halfinvvar1, kernel.k2 - cost * kernel.halfinvvar2)
robustify(kernel::ContaminatedGaussian, cost) = first(rhoresponsibility(cgterms(kernel, cost)...))
function robustifydcost(kernel::ContaminatedGaussian, cost)
    rho, gamma = rhoresponsibility(cgterms(kernel, cost)...)
    h1 = kernel.halfinvvar1
    h2 = kernel.halfinvvar2
    dh = h1 - h2
    return rho, 2 * (gamma * h1 + (1 - gamma) * h2), -2 * gamma * (1 - gamma) * dh * dh
end

function optimizekernel(kernel::ContaminatedGaussian{T, D}, squarederrors::Vector{T}, maxiters=10)::ContaminatedGaussian{T, D} where {T, D}
    # Optimize the parameters of the kernel using Expectation-Maximization algorithm
    totalsquarederror = sum(squarederrors)
    oldparams = params(kernel)
    for iter = 1:maxiters
        sigma1 = T(0)
        totalweight = T(0)
        for err in squarederrors
            # Expectation step - the first component's responsibility
            w = last(rhoresponsibility(cgterms(kernel, err)...))
            # Compute running totals for the Maximization step
            sigma1 += w * err
            totalweight += w
        end
        # Degenerate responsibilities (all in one component) leave the other component's scale undefined
        if !(totalweight > 0 && totalweight < length(squarederrors))
            break
        end
        # Complete the Maximization step (computing the optimal distribution parameters) by normalizing by the total weights
        newparams = SVector(sqrt(sigma1 / totalweight), sqrt((totalsquarederror - sigma1) / (length(squarederrors) - totalweight)), totalweight / length(squarederrors))
        kernel = ContaminatedGaussian(newparams..., static(D))
        if isapprox(oldparams, newparams; rtol=1.e-6)
            break
        end
        oldparams = newparams
    end
    return kernel
end
