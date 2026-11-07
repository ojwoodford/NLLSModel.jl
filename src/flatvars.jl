# Flat variable packing, unpacking, and layout helpers

@inline function pack_single_var!(x_flat::Vector{Float64}, x0::StaticVector{N}) where N
    @inbounds for k in 1:N
        x_flat[k] = Float64(x0[k])
    end
end
@inline pack_single_var!(x_flat::Vector{Float64}, x0::Number) = (@inbounds x_flat[1] = Float64(x0); nothing)
@inline function pack_single_var!(x_flat::Vector{Float64}, x0::AbstractVector)
    @inbounds for k in eachindex(x0)
        x_flat[k] = Float64(x0[k])
    end
end

@inline function pack_var!(x::Vector{Float64}, idx::Int, v::StaticVector{N}) where N
    @inbounds for k in 1:N
        x[idx + k - 1] = Float64(v[k])
    end
end
@inline pack_var!(x::Vector{Float64}, idx::Int, v::Number) = (@inbounds x[idx] = Float64(v); nothing)
@inline function pack_var!(x::Vector{Float64}, idx::Int, v::AbstractVector)
    @inbounds for k in eachindex(v)
        x[idx + k - 1] = Float64(v[k])
    end
end

copy_var(x::StaticVector) = x
copy_var(x::Number) = x
copy_var(x::Vector{T}) where T = copy(x)
copy_var(x) = deepcopy(x)

copy_vars_list(vars::Vector{T}) where T<:StaticVector = copy(vars)
copy_vars_list(vars::Vector{T}) where T<:Number = copy(vars)
copy_vars_list(vars::Vector) = deepcopy(vars)

@inline unpack_single_var(::StaticVector{N, T}, x) where {N, T} = SVector{N, T}(ntuple(k -> @inbounds(x[k]), Val(N)))
@inline unpack_single_var(::Number, x) = @inbounds x[1]
@inline function unpack_single_var(v::Vector{T}, x) where T
    copyto!(v, x)
    return v
end
@inline unpack_single_var(v, x) = typeof(v)(x)

@inline unpack_var_elem(::StaticVector{N, T}, x, idx, _) where {N, T} = SVector{N, T}(ntuple(k -> @inbounds(x[idx + k - 1]), Val(N)))
@inline unpack_var_elem(::Number, x, idx, _) = @inbounds x[idx]
@inline function unpack_var_elem(v::Vector{T}, x, idx, bsz) where T
    @inbounds for k in 1:bsz
        v[k] = x[idx + k - 1]
    end
    return v
end
@inline unpack_var_elem(v, x, idx, bsz) = typeof(v)(view(x, idx:idx+bsz-1))

struct FlatVarLayout{VarsBuf, BSZ, UInd}
    vars_buf::VarsBuf
    blocksizes::BSZ
    is_single_var::Bool
    unfixed_inds::UInd
    n::Int
end

function FlatVarLayout(problem::Model, unfixed=nothing)
    conv_unfixed = convertunfixed(unfixed, problem)
    if conv_unfixed isa Integer
        is_single = true
        unfixed_inds = conv_unfixed
    elseif count(conv_unfixed) == 1
        is_single = true
        unfixed_inds = findfirst(conv_unfixed)
    else
        is_single = false
        unfixed_inds = all(conv_unfixed) ? nothing : findall(conv_unfixed)
    end

    if is_single
        x0 = problem.variables[unfixed_inds]
        n = Int(dynamic(nvars(x0)))
        blocksizes = (n,)
        vars_buf = copy_var(x0)
    else
        x0 = unfixed_inds === nothing ? problem.variables : problem.variables[unfixed_inds]
        vars_buf = copy_vars_list(x0)
        blocksizes = [Int(dynamic(nvars(v))) for v in x0]
        n = sum(blocksizes)
    end
    return FlatVarLayout(vars_buf, blocksizes, is_single, unfixed_inds, n)
end

function pack_vars!(x_flat::Vector{Float64}, layout::FlatVarLayout, x0)
    if layout.is_single_var
        pack_single_var!(x_flat, x0)
    else
        idx = 1
        for (i, v) in enumerate(x0)
            pack_var!(x_flat, idx, v)
            idx += layout.blocksizes[i]
        end
    end
    return x_flat
end

function unpack_vars!(layout::FlatVarLayout, x_flat::AbstractVector{Float64})
    if layout.is_single_var
        return unpack_single_var(layout.vars_buf, x_flat)
    else
        idx = 1
        @inbounds for i in eachindex(layout.vars_buf)
            bsz = layout.blocksizes[i]
            layout.vars_buf[i] = unpack_var_elem(layout.vars_buf[i], x_flat, idx, bsz)
            idx += bsz
        end
        return layout.vars_buf
    end
end

function sync_flat_vars!(problem::Model, layout::FlatVarLayout, x_flat::AbstractVector{Float64})
    unpacked = unpack_vars!(layout, x_flat)
    if layout.is_single_var
        problem.variables[layout.unfixed_inds] = unpacked
    elseif layout.unfixed_inds === nothing
        copyto!(problem.variables, unpacked)
    else
        for (k, idx) in enumerate(layout.unfixed_inds)
            problem.variables[idx] = unpacked[k]
        end
    end
    return problem
end

# Trait to verify variables support Euclidean addition update(v, δ) == v + δ
isadditive(::Type{Float64}) = true
isadditive(::Type{<:StaticVector{N, Float64}}) where N = true
isadditive(::Type{Vector{Float64}}) = true
isadditive(::Type) = false
isadditive(v) = isadditive(typeof(v))

function check_additive_variable(v)
    if !isadditive(typeof(v))
        throw(ArgumentError("Variable of type $(typeof(v)) does not satisfy the `isadditive` trait (only Float64, SVector{N, Float64}, Vector{Float64} are supported for flat Euclidean updates)"))
    end
    nv = Int(dynamic(nvars(v)))
    xv = zeros(Float64, nv)
    pack_single_var!(xv, v)
    delta = [0.1 + 0.05 * k for k in 1:nv]
    v_upd = update(v, delta)
    x_upd = zeros(Float64, nv)
    pack_single_var!(x_upd, v_upd)
    if !isapprox(x_upd, xv + delta; rtol=1e-12, atol=1e-12)
        throw(ArgumentError("Variable of type $(typeof(v)) failed numerical additive check update(v, δ) ≈ v + δ"))
    end
    return true
end
