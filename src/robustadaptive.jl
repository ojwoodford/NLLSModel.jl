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
    return -2 * ((pos ? a : b) + log1p(e)), responsibility(pos, e)
end
# `γ` alone, from `t = a − b`, in the same form: `logistic(t)`
@inline responsibility(pos::Bool, e) = pos ? inv(1 + e) : e / (1 + e)
@inline function responsibility(t)
    pos = t >= 0
    return responsibility(pos, exp(pos ? -t : t))
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

struct GaussianPlusUniform{T<:Real, D} <: AbstractAdaptiveRobustifier
    invsigma::ZeroToInfScalar{T}   # u = invsigma.logval = log(1/σ)
    w::ZeroToOneScalar{T}          # the Gaussian's weight; v = w.logit
    halfinvvar::T                  # exp(2u)/2
    k::T                           # log(w) + D·u − (D/2)·log 2π
    log1mw::T                      # log(1 − w)
    logvolume::T                   # ℓ = log A, the outlier volume
end
function GaussianPlusUniform(invsigma::ZeroToInfScalar{T}, w::ZeroToOneScalar{T}, d::StaticInt, logvolume::Real) where T <: Real
    D = kerneldimension(d)
    u = invsigma.logval
    return GaussianPlusUniform{T, D}(invsigma, w, exp(2 * u) / 2, log(w.val) + D * u - halflog2pi(T, D), log(w.comp), convert(T, logvolume))
end
GaussianPlusUniform(invsigma::ZeroToInfScalar, w::ZeroToOneScalar, d::Integer, logvolume::Real) = kerneldimension(d)
function GaussianPlusUniform(sigma::Real, w::Real, d::Union{StaticInt, Integer}; volume::Real)
    T = float(promote_type(typeof(sigma), typeof(w)))
    kerneldimension(d)
    isfinite(volume) && volume > 0 || throw(ArgumentError("The outlier volume of GaussianPlusUniform must be finite and positive (got volume = $volume)"))
    return GaussianPlusUniform(ZeroToInfScalar{T}(inv(T(sigma))), ZeroToOneScalar{T}(T(w)), d, log(T(volume)))
end
nvars(::GaussianPlusUniform) = static(2)
update(var::GaussianPlusUniform{T, D}, updatevec, start=1) where {T, D} = GaussianPlusUniform(update(var.invsigma, updatevec, start), update(var.w, updatevec, start+1), static(D), var.logvolume)
params(var::GaussianPlusUniform) = SVector(1.0 / var.invsigma.val, var.w.val)
kerneldim(::GaussianPlusUniform{T, D}) where {T, D} = D
withlogvolume(kernel::GaussianPlusUniform{T, D}, logvolume::Real) where {T, D} =
    GaussianPlusUniform{T, D}(kernel.invsigma, kernel.w, kernel.halfinvvar, kernel.k, kernel.log1mw, convert(T, logvolume))

@inline gputerms(kernel::GaussianPlusUniform, cost) = (kernel.k - cost * kernel.halfinvvar, kernel.log1mw - kernel.logvolume)
robustify(kernel::GaussianPlusUniform, cost) = first(rhoresponsibility(gputerms(kernel, cost)...))
function robustifydcost(kernel::GaussianPlusUniform, cost)
    rho, gamma = rhoresponsibility(gputerms(kernel, cost)...)
    h = kernel.halfinvvar
    return rho, 2 * gamma * h, -2 * gamma * (1 - gamma) * h * h
end
# The derivatives over (u, v, s): u = log(1/σ), v = logit(w), s = ‖r‖²
function robustifydkernel(kernel::GaussianPlusUniform{T, D}, cost) where {T, D}
    rho, gamma = rhoresponsibility(gputerms(kernel, cost)...)
    iv = 2 * kernel.halfinvvar
    q = cost * iv
    p = gamma * (1 - gamma)
    wv = kernel.w.val
    Dq = D - q
    pDq = p * Dq
    r_u = -2 * gamma * Dq
    r_v = -2 * (gamma - wv)
    r_s = gamma * iv
    r_uu = 4 * gamma * q - 2 * pDq * Dq
    r_uv = -2 * pDq
    r_vv = 2 * (kernel.w.val * kernel.w.comp - p)
    r_us = (2 * gamma + pDq) * iv
    r_vs = p * iv
    r_ss = -p * iv * iv / 2
    return rho, SVector(r_u, r_v, r_s), SMatrix{3, 3}(r_uu, r_uv, r_us, r_uv, r_vv, r_vs, r_us, r_vs, r_ss)
end
