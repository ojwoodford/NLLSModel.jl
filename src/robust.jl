# Robustification
struct NoRobust <: AbstractRobustifier
end

robustkernel(::AbstractResidual) = NoRobust()
robustify(::NoRobust, cost) = cost
robustifydcost(::NoRobust, cost) = cost, one(cost), zero(cost)

robustifydcost(kernel::AbstractRobustifier, cost) = autorobustifydcost(kernel, cost)
robustifydkernel(kernel::AbstractAdaptiveRobustifier, cost) = autorobustifydkernel(kernel, cost)

struct Scaled{T<:Real,Robustifier<:AbstractRobustifier} <: AbstractRobustifier
    robust::Robustifier
    height::T
end
robustify(kernel::Scaled, cost) = robustify(kernel.robust, cost) * kernel.height
robustifydcost(kernel::Scaled{<:Real, NoRobust}, cost) = cost * kernel.height, kernel.height, zero(cost)
function robustifydcost(kernel::Scaled, cost)
    c, d1, d2 = robustifydcost(kernel.robust, cost)
    return c * kernel.height, d1 * kernel.height, d2 * kernel.height
end

islinearkernel(::NoRobust) = true
function islinearkernel(k::Scaled)
    h = Float64(dynamic(k.height))
    return islinearkernel(k.robust) && isfinite(h) && (h >= 0.0)
end
islinearkernel(::AbstractRobustifier) = false

linearweight(::NoRobust) = 1.0
linearweight(k::Scaled) = Float64(dynamic(k.height)) * linearweight(k.robust)

const DEFAULT_DC_TOL = 1e-12
@inline function kernel_dc_tol(kernel::AbstractRobustifier)::Float64
    _, dc0_raw, _ = robustifydcost(kernel, 0.0)
    dc0 = Float64(dynamic(dc0_raw))
    return (isfinite(dc0) && dc0 > 0.0) ? (DEFAULT_DC_TOL * dc0) : DEFAULT_DC_TOL
end

struct HuberKernel{T<:Real, B} <: AbstractRobustifier
    width::T
    width_squared::T
    secondorder::B
end
HuberKernel(w) = HuberKernel(w, w*w, static(false))
Huber2oKernel(w) = HuberKernel(w, w*w, static(true))

robustify(kernel::HuberKernel, cost) = cost < kernel.width_squared ? cost : sqrt(cost) * (kernel.width * 2) - kernel.width_squared
function robustifydcost(kernel::HuberKernel, cost)
    if cost < kernel.width_squared
        return cost, one(cost), zero(cost)
    end
    sqrtcost = sqrt(cost)
    return sqrtcost * (kernel.width * 2) - kernel.width_squared, kernel.width / sqrtcost, dynamic(kernel.secondorder) ? (-0.5 * kernel.width) / (cost * sqrtcost) : zero(cost)
end

struct GemanMcclureKernel{T<:Real, B} <: AbstractRobustifier
    width_squared::T
    secondorder::B
end
GemanMcclureKernel(w) = GemanMcclureKernel(w*w, static(false))
GemanMcclure2oKernel(w) = GemanMcclureKernel(w*w, static(true))

robustify(kernel::GemanMcclureKernel, cost) = cost * kernel.width_squared / (cost + kernel.width_squared)
function robustifydcost(kernel::GemanMcclureKernel, cost)
    r = 1.0 / (cost + kernel.width_squared)
    w = kernel.width_squared * r
    w2 = w * w
    return cost * w, w2, dynamic(kernel.secondorder) ? -2 * w2 * r : zero(cost)
end
