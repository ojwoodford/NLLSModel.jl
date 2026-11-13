using LoopVectorization: @turbo

# Native kernels for the gradient and Gauss-Newton Hessian of one residual (A1).
#
#     gramgrad!(H, g, J, r, dc, c2)
#
# computes, in one pass,
#
#     g = dc * (J' * r)
#     H = dc * (J' * J) + c2 * (J' * r) * (J' * r)'
#
# where `J` is the `M × n` Jacobian and `r` the length-`M` residual. `dc` is the first derivative of the robust kernel and
# `c2` its (positive semi-definite) second-order weight, `c2 = d2c == 0 ? 0 : max(2 * d2c, -dc / (3 * sqres))`.
#
# Contract
# - `H` is written as a FULL symmetric matrix (both triangles, `H == H'` exactly) and is completely overwritten, whatever
#   it held before. `updatesymA!` adds whole diagonal blocks and reads the upper triangle of off-diagonal blocks, so a
#   lower triangle alone is not enough. `g` is completely overwritten too.
# - No BLAS, no threads, no allocation. Everything done for one cost is single-threaded: costs are evaluated from
#   several threads at once, so any threading (BLAS included) inside one cost competes with the outer loop.
# - Wrong sizes throw `DimensionMismatch`.
#
# Kernels (the full static multiply, kernel 1, is `gramgrad` below, which works on static arrays and returns static arrays).
# For the dynamic ones, `gram_kernel` / `jtr_kernel` choose in this order:
# 2. static-column kernel: `J` is a `Matrix` of `Float32`/`Float64` and `r` an `SVector{M}` with `M <= MAX_STATIC_ROWS`;
#    `J` is viewed as `n` static columns and the dot products are unrolled.
# 3. plain loops when `J` has a single column (`n == 1`), whatever the types: a column of many rows is a single dot
#    product, which the plain loop does 2 to 4 times faster than `@turbo` (benchmark/results).
# 4. `@turbo` kernel: every array is a strided array of the same hardware float type (`Array`s and unit-range views of
#    them): a `Vector` residual with a `Matrix` Jacobian, for instance. `@turbo` needs arrays with a memory address.
# 5. plain loops (`@fastmath @inbounds @simd`) for anything else: an `Adjoint`, a non-contiguous view, another element type, and
#    an `SVector` longer than `MAX_STATIC_ROWS` or an `MVector` of any length with a `Matrix` Jacobian. These are the
#    intended path, not a gap: nothing is copied and `@turbo` is not offered the types it cannot address.
# `@fastmath` lets the compiler assume finite values, so the plain kernels are tested to still carry a `NaN` or `Inf`
# in `J` or `r` through to `g` and `H` (the optimizer terminates on a non-finite gradient).

const GramFloat = Union{Float32, Float64}
const _GramIdx = Union{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}
# Arrays `@turbo` can address: dense arrays and views of them with unit-range indices
const StridedGram{T, N} = Union{Array{T, N}, SubArray{T, N, <:Array{T}, <:Tuple{Vararg{_GramIdx}}}}

turbo_ok(::Type, ::Type, ::Type, ::Type) = false
turbo_ok(::Type{<:StridedGram{T, 2}}, ::Type{<:StridedGram{T, 1}}, ::Type{<:StridedGram{T, 2}}, ::Type{<:StridedGram{T, 1}}) where {T <: GramFloat} = true

staticcols_ok(::Type, ::Type) = false
staticcols_ok(::Type{Matrix{T}}, ::Type{<:SVector{M, T}}) where {T <: GramFloat, M} = M <= MAX_STATIC_ROWS

@noinline function gram_size_error(H, g, J, r)
    throw(DimensionMismatch("gramgrad!: J is $(size(J, 1))x$(size(J, 2)), r has length $(length(r)), g has length $(length(g)), H is $(size(H, 1))x$(size(H, 2))"))
end

@inline function gram_check(H, g, J, r)
    M, n = size(J)
    (length(r) == M && length(g) == n && size(H, 1) == n && size(H, 2) == n) || gram_size_error(H, g, J, r)
    return M, n
end

# Which dynamic kernel runs: :staticcols, :turbo or :plain (see the contract at the top)
@inline function gram_kernel(H, g, J, r)
    staticcols_ok(typeof(J), typeof(r)) && return :staticcols
    size(J, 2) == 1 && return :plain
    turbo_ok(typeof(H), typeof(g), typeof(J), typeof(r)) && return :turbo
    return :plain
end
@inline jtr_kernel(g, J, r) = gram_kernel(J, g, J, r) # (`J` stands in for the 2-D `H` that the predicate wants)

# g = J' * r, unweighted
function jtr!(g, J, r)
    M, n = size(J)
    (length(r) == M && length(g) == n) || throw(DimensionMismatch("jtr!: J is $(M)x$n, r has length $(length(r)), g has length $(length(g))"))
    kernel = jtr_kernel(g, J, r)
    if kernel === :staticcols
        jtr_staticcols!(g, J, r)
    elseif kernel === :turbo
        jtr_turbo!(g, J, r)
    else
        jtr_plain!(g, J, r)
    end
    return g
end

function gramgrad!(H, g, J, r, dc, c2)
    gram_check(H, g, J, r)
    T = eltype(H)
    dcT, c2T = convert(T, dc), convert(T, c2)
    kernel = gram_kernel(H, g, J, r)
    if kernel === :staticcols
        gram_staticcols!(H, g, J, r, dcT, c2T)
    elseif kernel === :turbo
        gram_turbo!(H, g, J, r, dcT, c2T)
    else
        gram_plain!(H, g, J, r, dcT, c2T)
    end
    return H
end

# Scalar residual: J is the transposed gradient and r a number, so J'J = ∇∇' and J'r = r∇
function gramgrad!(H, g, J::Adjoint{<:Any, <:AbstractVector}, r::Number, dc, c2)
    ∇ = parent(J)
    n = length(∇)
    (length(g) == n && size(H, 1) == n && size(H, 2) == n) || gram_size_error(H, g, J, r)
    T = eltype(H)
    s = convert(T, dc + c2 * r * r)
    f = convert(T, dc * r)
    @inbounds for i in 1:n
        g[i] = f * ∇[i]
        si = s * ∇[i]
        for j in 1:i
            v = si * ∇[j]
            H[i, j] = v
            H[j, i] = v
        end
    end
    return H
end

# Mirror the weighted lower triangle: H[i, j] = dc * H[i, j] + c2 * gu[i] * gu[j], written to both triangles
@inline function gram_weight_mirror!(H, gu, n, dc, c2)
    @inbounds for i in 1:n
        gi = c2 * gu[i]
        for j in 1:i
            v = muladd(gi, gu[j], dc * H[i, j])
            H[i, j] = v
            H[j, i] = v
        end
    end
end

@inline function gram_scale!(g, n, dc)
    @inbounds for i in 1:n
        g[i] *= dc
    end
end

# ---- Plain loops ----
function jtr_plain!(g, J, r)
    M, n = size(J)
    @inbounds for i in 1:n
        acc = zero(eltype(g))
        @fastmath @simd for k in 1:M
            acc += J[k, i] * r[k]
        end
        g[i] = acc
    end
end

function gram_plain!(H, g, J, r, dc, c2)
    M, n = size(J)
    jtr_plain!(g, J, r)
    @inbounds for i in 1:n
        for j in 1:i
            acc = zero(eltype(H))
            @fastmath @simd for k in 1:M
                acc += J[k, i] * J[k, j]
            end
            H[i, j] = acc
        end
    end
    gram_weight_mirror!(H, g, n, dc, c2)
    gram_scale!(g, n, dc)
end

# ---- @turbo ----
function jtr_turbo!(g, J, r)
    M, n = size(J)
    @turbo for i in 1:n
        acc = zero(eltype(g))
        for k in 1:M
            acc += J[k, i] * r[k]
        end
        g[i] = acc
    end
end

function gram_turbo!(H, g, J, r, dc, c2)
    M, n = size(J)
    jtr_turbo!(g, J, r)
    for i in 1:n
        @turbo for j in 1:i
            acc = zero(eltype(H))
            for k in 1:M
                acc += J[k, i] * J[k, j]
            end
            H[i, j] = acc
        end
    end
    gram_weight_mirror!(H, g, n, dc, c2)
    gram_scale!(g, n, dc)
end

# ---- Static columns ----
@inline staticcols(J::Matrix{T}, ::SVector{M, T}) where {T, M} = reinterpret(reshape, SVector{M, T}, J)

function jtr_staticcols!(g, J, r)
    cols = staticcols(J, r)
    @inbounds for i in 1:size(J, 2)
        g[i] = dot(cols[i], r)
    end
end

function gram_staticcols!(H, g, J, r, dc, c2)
    cols = staticcols(J, r)
    n = size(J, 2)
    jtr_staticcols!(g, J, r)
    @inbounds for i in 1:n
        ci = cols[i]
        gi = c2 * g[i]
        for j in 1:i
            v = muladd(gi, g[j], dc * dot(ci, cols[j]))
            H[i, j] = v
            H[j, i] = v
        end
    end
    gram_scale!(g, n, dc)
end

# ---- Kernel 1: full static multiply ----
# Same quantities for a static Jacobian, returning static arrays (the compiler unrolls and fuses everything)
function gramgrad(J, r, dc, c2)
    g = J' * r
    H = J' * J
    if dc != 1
        H *= dc
    end
    if c2 != 0
        H += (c2 * g) * g'
    end
    if dc != 1
        g *= dc
    end
    return g, H
end
