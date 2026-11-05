module NLPModelExt

using NLLSModels
using NLPModels
using LinearAlgebra
using SparseArrays
using StaticArrays
using Static
using OptimizationVariableInterface
import OptimizationVariableInterface: nvars, syncvars!

# Packing helpers
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

# Copy helpers
copy_var(x::StaticVector) = x
copy_var(x::Number) = x
copy_var(x::Vector{T}) where T = copy(x)
copy_var(x) = deepcopy(x)

copy_vars_list(vars::Vector{T}) where T<:StaticVector = copy(vars)
copy_vars_list(vars::Vector{T}) where T<:Number = copy(vars)
copy_vars_list(vars::Vector) = deepcopy(vars)

# Unpacking helpers
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

"""
    NLLSModelNLP <: AbstractNLPModel{Float64, Vector{Float64}}

Concrete wrapper adapting an `NLLSProblem` to the `NLPModels.AbstractNLPModel` interface.
Enables solving nonlinear least-squares problems formulated in `NLLSModels` using general
NLP solvers including `JSOSolvers`, `Ipopt`, and `Optim`.
"""
mutable struct NLLSModelNLP{P, F, FDF, X0, H, VarsBuf, BSZ, HIdx, UInd} <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    problem::P
    f::F
    fdf!::FDF
    x0::X0
    g_buf::Vector{Float64}
    H_buf::H
    vars_buf::VarsBuf
    blocksizes::BSZ
    is_single_var::Bool
    unfixed_inds::UInd
    h_rows::Vector{Int}
    h_cols::Vector{Int}
    h_nzindices::HIdx
    # Caching
    last_x::Vector{Float64}
    last_cost::Base.RefValue{Float64}
    cache_valid::Base.RefValue{Bool}
end

function NLLSModels.NLLSModel(problem::NLLSProblem; unfixed=nothing, numthreads=static(1), name="NLLSModel")
    f, fdf!, x0, H_temp = optimvars(problem, unfixed, numthreads)
    conv_unfixed = NLLSModels.convertunfixed(unfixed, problem)

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
        n = Int(dynamic(nvars(x0)))
        blocksizes = (n,)
        vars_buf = copy_var(x0)
        x0_flat = Vector{Float64}(undef, n)
        pack_single_var!(x0_flat, x0)
        H_buf = Matrix{Float64}(undef, n, n)
        nnzh = n * (n + 1) ÷ 2
        h_rows = Vector{Int}(undef, nnzh)
        h_cols = Vector{Int}(undef, nnzh)
        idx = 1
        for j in 1:n
            for i in j:n
                h_rows[idx] = i
                h_cols[idx] = j
                idx += 1
            end
        end
        h_nzindices = nothing
    else
        vars_buf = copy_vars_list(x0)
        blocksizes = [Int(dynamic(nvars(v))) for v in x0]
        n = sum(blocksizes)
        x0_flat = Vector{Float64}(undef, n)
        idx = 1
        for (i, v) in enumerate(x0)
            pack_var!(x0_flat, idx, v)
            idx += blocksizes[i]
        end
        if H_temp === nothing
            H_buf = Matrix{Float64}(undef, n, n)
            nnzh = n * (n + 1) ÷ 2
            h_rows = Vector{Int}(undef, nnzh)
            h_cols = Vector{Int}(undef, nnzh)
            idx = 1
            for j in 1:n
                for i in j:n
                    h_rows[idx] = i
                    h_cols[idx] = j
                    idx += 1
                end
            end
            h_nzindices = nothing
        else
            H_buf = copy(H_temp)
            h_rows = Int[]
            h_cols = Int[]
            h_nzindices = Int[]
            for j in 1:n
                for p in H_buf.colptr[j]:(H_buf.colptr[j+1]-1)
                    i = H_buf.rowval[p]
                    if i >= j
                        push!(h_rows, i)
                        push!(h_cols, j)
                        push!(h_nzindices, p)
                    end
                end
            end
            nnzh = length(h_rows)
        end
    end

    meta = NLPModelMeta(n, x0=copy(x0_flat), nnzh=nnzh, name=name)
    g_buf = Vector{Float64}(undef, n)
    last_x = fill(NaN, n)

    return NLLSModelNLP(
        meta,
        Counters(),
        problem,
        f,
        fdf!,
        x0,
        g_buf,
        H_buf,
        vars_buf,
        blocksizes,
        is_single,
        unfixed_inds,
        h_rows,
        h_cols,
        h_nzindices,
        last_x,
        Ref(0.0),
        Ref(false)
    )
end

NLPModels.AbstractNLPModel(problem::NLLSProblem; kwargs...) = NLLSModels.NLLSModel(problem; kwargs...)
Base.convert(::Type{AbstractNLPModel}, problem::NLLSProblem) = NLLSModels.NLLSModel(problem)

function unpack_to_vars!(nlp::NLLSModelNLP, x::AbstractVector)
    if nlp.is_single_var
        return unpack_single_var(nlp.vars_buf, x)
    else
        idx = 1
        @inbounds for i in eachindex(nlp.vars_buf)
            bsz = nlp.blocksizes[i]
            nlp.vars_buf[i] = unpack_var_elem(nlp.vars_buf[i], x, idx, bsz)
            idx += bsz
        end
        return nlp.vars_buf
    end
end

function ensure_fdf!(nlp::NLLSModelNLP, x::AbstractVector)
    if !nlp.cache_valid[] || x != nlp.last_x
        copyto!(nlp.last_x, x)
        unpacked = unpack_to_vars!(nlp, x)
        nlp.last_cost[] = nlp.fdf!(nlp.g_buf, nlp.H_buf, unpacked)
        nlp.cache_valid[] = true
    end
    return nlp.last_cost[]
end

# Objective
function NLPModels.obj(nlp::NLLSModelNLP, x::AbstractVector)
    increment!(nlp, :neval_obj)
    if nlp.cache_valid[] && x == nlp.last_x
        return nlp.last_cost[]
    end
    unpacked = unpack_to_vars!(nlp, x)
    return nlp.f(unpacked)
end

# Gradient
function NLPModels.grad!(nlp::NLLSModelNLP, x::AbstractVector, g::AbstractVector)
    increment!(nlp, :neval_grad)
    ensure_fdf!(nlp, x)
    copyto!(g, nlp.g_buf)
    return g
end

# Objective and gradient
function NLPModels.objgrad!(nlp::NLLSModelNLP, x::AbstractVector, g::AbstractVector)
    increment!(nlp, :neval_obj)
    increment!(nlp, :neval_grad)
    c = ensure_fdf!(nlp, x)
    copyto!(g, nlp.g_buf)
    return c, g
end

# Hessian structure (lower triangular)
function NLPModels.hess_structure!(nlp::NLLSModelNLP, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    copyto!(rows, nlp.h_rows)
    copyto!(cols, nlp.h_cols)
    return rows, cols
end

# Hessian coordinates (lower triangular)
function NLPModels.hess_coord!(nlp::NLLSModelNLP, x::AbstractVector, vals::AbstractVector; obj_weight::Real=1.0)
    increment!(nlp, :neval_hess)
    ensure_fdf!(nlp, x)
    w = Float64(obj_weight)
    if nlp.h_nzindices === nothing
        @inbounds for k in 1:length(nlp.h_rows)
            vals[k] = w * nlp.H_buf[nlp.h_rows[k], nlp.h_cols[k]]
        end
    else
        @inbounds for k in 1:length(nlp.h_nzindices)
            vals[k] = w * nlp.H_buf.nzval[nlp.h_nzindices[k]]
        end
    end
    return vals
end

function NLPModels.hess_coord!(nlp::NLLSModelNLP, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight::Real=1.0)
    return NLPModels.hess_coord!(nlp, x, vals; obj_weight=obj_weight)
end

# Hessian-vector product
function NLPModels.hprod!(nlp::NLLSModelNLP, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real=1.0)
    increment!(nlp, :neval_hprod)
    ensure_fdf!(nlp, x)
    mul!(Hv, Symmetric(nlp.H_buf, :L), v)
    w = Float64(obj_weight)
    if w != 1.0
        Hv .*= w
    end
    return Hv
end

function NLPModels.hprod!(nlp::NLLSModelNLP, x::AbstractVector, y::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real=1.0)
    return NLPModels.hprod!(nlp, x, v, Hv; obj_weight=obj_weight)
end

# Reset counters & cache
function NLPModels.reset!(nlp::NLLSModelNLP)
    reset!(nlp.counters)
    nlp.cache_valid[] = false
    fill!(nlp.last_x, NaN)
    return nlp
end

# Synchronize solution back to problem variables
"""
    syncvars!(nlp::NLLSModelNLP, x::AbstractVector) -> NLLSProblem
    syncvars!(prob::NLLSProblem, nlp::NLLSModelNLP, x::AbstractVector) -> NLLSProblem

Copies the flat variable vector `x` back into the wrapped problem's `variables` array.
"""
function OptimizationVariableInterface.syncvars!(nlp::NLLSModelNLP, x::AbstractVector)
    unpacked = unpack_to_vars!(nlp, x)
    if nlp.is_single_var
        nlp.problem.variables[nlp.unfixed_inds] = unpacked
    elseif nlp.unfixed_inds === nothing
        copyto!(nlp.problem.variables, unpacked)
    else
        for (k, idx) in enumerate(nlp.unfixed_inds)
            nlp.problem.variables[idx] = unpacked[k]
        end
    end
    return nlp.problem
end

OptimizationVariableInterface.syncvars!(::NLLSProblem, nlp::NLLSModelNLP, x::AbstractVector) =
    syncvars!(nlp, x)

end # module NLPModelExt
