using Static

# Simple residuals modelling measurement noise errors
struct SimpleError2{N, T, V1, V2} <: AbstractResidual
    measurement::SVector{N, T} # Measurement
    varind::SVector{2, Int} # Variable indices
end
SimpleError2{V1, V2}(meas::SVector{N, T}, vi1, vi2) where {N, T, V1, V2} = SimpleError2{N, T, V1, V2}(meas, SVector{2, Int}(vi1, vi2))
ndeps(::SimpleError2) = static(2) # Residual depends on 2 variables
nres(::SimpleError2{N, T, V1, V2}) where {N, T, V1, V2} = static(N) # Residual vector has length N
varindices(res::SimpleError2) = res.varind
getvars(res::SimpleError2{N, T, V1, V2}, vars::AbstractVector) where {N, T, V1, V2} = vars[@inbounds(res.varind[1])]::V1, vars[@inbounds(res.varind[2])]::V2
computeresidual(res::SimpleError2, arg1, arg2) = generatemeasurement(arg1, arg2) - res.measurement
Base.eltype(::SimpleError2{N, T, V1, V2}) where {N, T, V1, V2} = T

struct SimpleError3{N, T, V1, V2, V3} <: AbstractResidual
    measurement::SVector{N, T} # Measurement
    varind::SVector{3, Int} # Variable indices
end
SimpleError3{V1, V2, V3}(meas::SVector{N, T}, vi1, vi2, vi3) where {N, T, V1, V2, V3} = SimpleError3{N, T, V1, V2, V3}(meas, SVector{3, Int}(vi1, vi2, vi3))
ndeps(::SimpleError3) = static(3) # Residual depends on 3 variables
nres(::SimpleError3{N, T, V1, V2, V3}) where {N, T, V1, V2, V3} = static(N) # Residual vector has length N
varindices(res::SimpleError3) = res.varind
getvars(res::SimpleError3{N, T, V1, V2, V3}, vars::AbstractVector) where {N, T, V1, V2, V3} = vars[@inbounds(res.varind[1])]::V1, vars[@inbounds(res.varind[2])]::V2, vars[@inbounds(res.varind[3])]::V3
computeresidual(res::SimpleError3, arg1, arg2, arg3) = generatemeasurement(arg1, arg2, arg3) - res.measurement
Base.eltype(::SimpleError3{N, T, V1, V2, V3}) where {N, T, V1, V2, V3} = T

struct SimpleError4{N, T, V1, V2, V3, V4} <: AbstractResidual
    measurement::SVector{N, T} # Measurement
    varind::SVector{4, Int} # Variable indices
end
SimpleError4{V1, V2, V3, V4}(meas::SVector{N, T}, vi1, vi2, vi3, vi4) where {N, T, V1, V2, V3, V4} = SimpleError4{N, T, V1, V2, V3, V4}(meas, SVector{4, Int}(vi1, vi2, vi3, vi4))
ndeps(::SimpleError4) = static(4) # Residual depends on 4 variables
nres(::SimpleError4{N, T, V1, V2, V3, V4}) where {N, T, V1, V2, V3, V4} = static(N) # Residual vector has length N
varindices(res::SimpleError4) = res.varind
getvars(res::SimpleError4{N, T, V1, V2, V3, V4}, vars::AbstractVector) where {N, T, V1, V2, V3, V4} = vars[@inbounds(res.varind[1])]::V1, vars[@inbounds(res.varind[2])]::V2, vars[@inbounds(res.varind[3])]::V3, vars[@inbounds(res.varind[4])]::V4
computeresidual(res::SimpleError4, arg1, arg2, arg3, arg4) = generatemeasurement(arg1, arg2, arg3, arg4) - res.measurement
Base.eltype(::SimpleError4{N, T, V1, V2, V3, V4}) where {N, T, V1, V2, V3, V4} = T

# Dummy definition
generatemeasurement() = nothing

# Functions for computing costs and gradients of residuals
computecost(residual::AbstractResidual, vars...) = computerescost(residual, robustkernel(residual), vars)
computecostgradhess(varflags, residual::AbstractResidual, vars...) = computerescostgradhess(varflags << static(1), residual, robustkernel(residual), vars)
computecost(residual::AbstractAdaptiveResidual, kernel, vars...) = computerescost(residual, kernel, vars)
computecostgradhess(varflags, residual::AbstractAdaptiveResidual, kernel, vars...) = computerescostgradhess(varflags, residual, kernel, vars)

function computerescost(residual, kernel, vars)::Float64
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return 0.5 * robustify(kernel, Float64(sqnorm(r)))
end

# Second-order weight of the robust kernel, positive semi-definite by construction
gram_c2(dc, d2c, sqres) = d2c == 0 ? zero(dc) : max(2 * d2c, -dc / (3 * sqres))

# Whether a Jacobian is a static array (so the whole computation is unrolled static arithmetic)
isstaticjac(::StaticMatrix) = true
isstaticjac(::Adjoint{<:Any, <:StaticVector}) = true
isstaticjac(_) = false

# Per-task buffers for the gradient and Hessian of one cost, sized for the kernel parameters plus the variables, so
# that nothing is allocated per cost. `g` and `H` returned from `computerescostgradhess` alias these: they are valid
# until the same task evaluates its next cost (`gradhesshelper!` adds them to the linear system straight away).
struct GramBuffers{T}
    g::Vector{T}   # Kernel parameters, then variables
    H::Matrix{T}
    gu::Vector{T}  # Unweighted J'r (needed for the kernel-parameter cross terms of adaptive kernels)
end

function gram_buffers(::Type{T}, nk::Int, n::Int)::GramBuffers{T} where T
    cache = get!(() -> Dict{DataType, Any}(), task_local_storage(), :nllsmodel_gramcache)::Dict{DataType, Any}
    bycost = get!(() -> Dict{Tuple{Int, Int}, GramBuffers{T}}(), cache, T)::Dict{Tuple{Int, Int}, GramBuffers{T}}
    return get!(() -> GramBuffers{T}(zeros(T, nk + n), zeros(T, nk + n, nk + n), zeros(T, n)), bycost, (nk, n))
end

# Residual as an SVector when its length is static and small, so the static-column kernel can be used
@inline staticresidual(res::Number, residual) = res
@inline staticresidual(res, residual) = staticresidual(res, nres(residual))
@inline staticresidual(res, ::Integer) = res
@inline function staticresidual(res::Vector{T}, ::StaticInt{M}) where {T, M}
    M <= MAX_STATIC_ROWS || return res
    return SVector{M, T}(ntuple(i -> @inbounds(res[i]), Val(M)))
end

function computerescostgradhess(varflags, residual, kernel, vars)
    # Check for case that only kernel is optimized
    varflagsres = varflags >> static(1)
    if varflagsres == 0
        # Only the kernel is optimized
        res = computeresidual(residual, vars...)
        cost, dc, d2c = robustifydkernel(kernel, sqnorm(res))
        kernelvarind = SR(1, nvars(kernel))
        # The cost is 0.5 * rho, so the kernel derivatives are halved
        return @inbounds (0.5 * Float64(cost), 0.5 * dc[kernelvarind], 0.5 * d2c[kernelvarind, kernelvarind])
    end

    # Compute the residual and Jacobian
    res, jac = computeresjac(varflagsres, residual, vars...)
    sqres = sqnorm(res)
    if isstaticjac(jac)
        return computerescostgradhess_static(varflags, kernel, res, jac, sqres)
    end
    return computerescostgradhess_dynamic(varflags, kernel, staticresidual(res, residual), jac, sqres)
end

# Static Jacobians: the full static multiply (kernel 1), unrolled by the compiler
function computerescostgradhess_static(varflags, kernel, res, jac, sqres)
    if varflags & 1 == 0
        # The kernel is not optimized. Differentiate w.r.t. the cost only
        cost, dc, d2c = robustifydcost(kernel, sqres)
        g, H = gramgrad(jac, res, dc, gram_c2(dc, d2c, sqres))
        return 0.5 * Float64(cost), g, H
    end

    # Differentiate w.r.t. the cost and kernel parameters
    cost, dc_, d2c_ = robustifydkernel(kernel, sqres)
    @inbounds dc = dc_[end]
    @inbounds d2c = d2c_[end,end]
    g, H = gramgrad(jac, res, dc, gram_c2(dc, d2c, sqres))

    # Compute the d^2/dkernel.dvariables block (from the unweighted gradient) and add on the kernel derivative blocks
    kernelvarind = SR(1, nvars(kernel))
    @inbounds dkdv = (jac' * res) * view(d2c_, kernelvarind, nvars(kernel)+1)'
    @inbounds g = vcat(0.5 * view(dc_, kernelvarind), g)
    @inbounds H = hcat(vcat(0.5 * view(d2c_, kernelvarind, kernelvarind), dkdv), vcat(dkdv', H))
    return 0.5 * Float64(cost), g, H
end

# Dynamic Jacobians: `gramgrad!` into per-task buffers (the kernel is evaluated before the products)
function computerescostgradhess_dynamic(varflags, kernel, res, jac, sqres)
    n = size(jac, 2)
    T = eltype(jac)
    if varflags & 1 == 0
        # The kernel is not optimized. Differentiate w.r.t. the cost only
        cost, dc, d2c = robustifydcost(kernel, sqres)
        b = gram_buffers(T, 0, n)
        gramgrad!(b.H, b.g, jac, res, dc, gram_c2(dc, d2c, sqres))
        return 0.5 * Float64(cost), b.g, b.H
    end

    # Differentiate w.r.t. the cost and kernel parameters
    cost, dc_, d2c_ = robustifydkernel(kernel, sqres)
    @inbounds dc = dc_[end]
    @inbounds d2c = d2c_[end,end]
    nk = dynamic(nvars(kernel))
    b = gram_buffers(T, nk, n)
    vars_ = nk+1:nk+n
    gramgrad!(view(b.H, vars_, vars_), view(b.g, vars_), jac, res, dc, gram_c2(dc, d2c, sqres))
    jtr!(b.gu, jac, res) # The unweighted gradient, for the d^2/dkernel.dvariables block
    @inbounds for k in 1:nk
        b.g[k] = 0.5 * dc_[k]
        ck = d2c_[k, nk+1]
        for l in 1:nk
            b.H[k, l] = 0.5 * d2c_[k, l]
        end
        for j in 1:n
            v = ck * b.gu[j]
            b.H[k, nk+j] = v
            b.H[nk+j, k] = v
        end
    end
    return 0.5 * Float64(cost), b.g, b.H
end
