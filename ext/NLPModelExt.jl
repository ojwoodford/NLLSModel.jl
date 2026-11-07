module NLPModelExt

using NLLSModel
using NLPModels
using LinearAlgebra
using SparseArrays
using StaticArrays
using Static
using OptimizationVariableInterface
import OptimizationVariableInterface: nvars, syncvars!
import NLLSModel: islinearkernel, flat_optimvars_nls, sync_flat_vars!

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

Concrete wrapper adapting an `NLLSModel.Model` to the `NLPModels.AbstractNLPModel` interface.
Enables solving nonlinear least-squares problems formulated in `NLLSModel` using general
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

function NLLSModel.NLPLSModel(problem::Model; unfixed=nothing, numthreads=static(1), name="NLPLSModel")
    f, fdf!, x0, H_temp = optimvars(problem, unfixed, numthreads)
    conv_unfixed = NLLSModel.convertunfixed(unfixed, problem)

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

NLPModels.AbstractNLPModel(problem::Model; kwargs...) = NLLSModel.NLPLSModel(problem; kwargs...)
Base.convert(::Type{AbstractNLPModel}, problem::Model) = NLLSModel.NLPLSModel(problem)

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
    syncvars!(nlp::NLLSModelNLP, x::AbstractVector) -> Model
    syncvars!(prob::Model, nlp::NLLSModelNLP, x::AbstractVector) -> Model

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

OptimizationVariableInterface.syncvars!(::Model, nlp::NLLSModelNLP, x::AbstractVector) =
    syncvars!(nlp, x)

"""
    NLLSModelNLS <: AbstractNLSModel{Float64, Vector{Float64}}

Concrete adapter exposing an `NLLSModel.Model` as an `NLPModels.AbstractNLSModel`.
Enables native tall NLS solvers such as `JSOSolvers.trunk` and `JSOSolvers.tron`.
Restricted to linear robust kernels (`islinearkernel`).
"""
mutable struct NLLSModelNLS{P, L, RES, RESJAC, J} <: AbstractNLSModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    nls_meta::NLSMeta{Float64, Vector{Float64}}
    counters::NLSCounters
    problem::P
    layout::L
    res_flat!::RES
    resjac_flat!::RESJAC
    r_buf::Vector{Float64}
    J_buf::J
    x_cached::Vector{Float64}
    last_cost::Float64
    res_valid::Bool
    jac_valid::Bool
end

"""
    NLLSNLSModel(problem::Model; unfixed=nothing, numthreads=static(1), min_costs_per_thread=1000, matrix_type=:auto, max_dense_bytes=32*1024^2, name="NLLSNLSModel")

Constructs an `NLLSModelNLS <: AbstractNLSModel{Float64, Vector{Float64}}` adapter for `problem`.
Requires all costs to be non-adaptive `AbstractResidual`s with linear robust kernels (`islinearkernel`).
Defaults `max_dense_bytes` to 32 MB (`32 * 1024^2`) to prevent solver-side coordinate arrays from exceeding ~96 MB.
"""
function NLLSModel.NLLSNLSModel(
    problem::Model;
    unfixed = nothing,
    numthreads = static(1),
    min_costs_per_thread::Int = 1000,
    matrix_type::Symbol = :auto,
    max_dense_bytes::Int = 32 * 1024^2,
    name::String = "NLLSNLSModel",
)
    for costvec in values(problem.costs)
        for c in costvec
            if c isa AbstractAdaptiveResidual || !islinearkernel(robustkernel(c))
                throw(ArgumentError("NLLSNLSModel only supports linear kernels (NoRobust or finite non-negative Scaled linear kernels) on non-adaptive residuals; found $(typeof(c isa AbstractAdaptiveResidual ? c : robustkernel(c)))"))
            end
        end
    end

    robust_mode = all(c -> robustkernel(c) isa NoRobust, Iterators.flatten(values(problem.costs))) ? :none : :irls
    flat = flat_optimvars_nls(
        problem;
        unfixed = unfixed,
        numthreads = numthreads,
        min_costs_per_thread = min_costs_per_thread,
        robust = robust_mode,
        matrix_type = matrix_type,
        max_dense_bytes = max_dense_bytes,
    )

    M = flat.layout_tall.M
    N = flat.layout.n
    nnzj = flat.J_proto isa SparseMatrixCSC ? length(flat.J_proto.nzval) : (M * N)

    meta = NLPModelMeta(N; x0 = copy(flat.x0), nnzh = 0, hess_available = false, name = name)
    nls_meta = NLSMeta(M, N; x0 = copy(flat.x0), nnzj = nnzj, nnzh = 0, hess_residual_available = false, hprod_residual_available = false)
    counters = NLSCounters()
    r_buf = Vector{Float64}(undef, M)
    J_buf = copy(flat.J_proto)
    x_cached = fill(NaN, N)

    return NLLSModelNLS(
        meta,
        nls_meta,
        counters,
        problem,
        flat.layout,
        flat.res!,
        flat.resjac!,
        r_buf,
        J_buf,
        x_cached,
        0.0,
        false,
        false,
    )
end

function invalidate_if_moved!(nls::NLLSModelNLS, x::AbstractVector)
    if x != nls.x_cached
        copyto!(nls.x_cached, x)
        nls.res_valid = false
        nls.jac_valid = false
    end
end

function ensure_res!(nls::NLLSModelNLS, x::AbstractVector)
    invalidate_if_moved!(nls, x)
    if !nls.res_valid
        nls.last_cost = nls.res_flat!(nls.r_buf, x)
        nls.res_valid = true
    end
    return nls.last_cost
end

function ensure_jac!(nls::NLLSModelNLS, x::AbstractVector)
    invalidate_if_moved!(nls, x)
    if !nls.jac_valid
        nls.last_cost = nls.resjac_flat!(nls.r_buf, nls.J_buf, x; prezeroed=true)
        nls.res_valid = true
        nls.jac_valid = true
    end
    return nls.last_cost
end

function NLPModels.residual!(nls::NLLSModelNLS, x::AbstractVector, F::AbstractVector)
    NLPModels.increment!(nls, :neval_residual)
    ensure_res!(nls, x)
    copyto!(F, nls.r_buf)
    return F
end

function NLPModels.jac_structure_residual!(nls::NLLSModelNLS, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    if nls.J_buf isa SparseMatrixCSC
        copyto!(rows, nls.J_buf.rowval)
        colptr = nls.J_buf.colptr
        N = nls.meta.nvar
        for j in 1:N
            for idx in colptr[j]:(colptr[j+1] - 1)
                cols[idx] = j
            end
        end
    else
        M = nls.nls_meta.nequ
        N = nls.meta.nvar
        idx = 1
        for j in 1:N
            for i in 1:M
                rows[idx] = i
                cols[idx] = j
                idx += 1
            end
        end
    end
    return rows, cols
end

function NLPModels.jac_coord_residual!(nls::NLLSModelNLS, x::AbstractVector, vals::AbstractVector)
    NLPModels.increment!(nls, :neval_jac_residual)
    ensure_jac!(nls, x)
    if nls.J_buf isa SparseMatrixCSC
        copyto!(vals, nls.J_buf.nzval)
    else
        copyto!(vals, nls.J_buf)
    end
    return vals
end

function NLPModels.jprod_residual!(nls::NLLSModelNLS, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    NLPModels.increment!(nls, :neval_jprod_residual)
    ensure_jac!(nls, x)
    mul!(Jv, nls.J_buf, v)
    return Jv
end

function NLPModels.jtprod_residual!(nls::NLLSModelNLS, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    NLPModels.increment!(nls, :neval_jtprod_residual)
    ensure_jac!(nls, x)
    mul!(Jtv, transpose(nls.J_buf), v)
    return Jtv
end

function NLPModels.obj(nls::NLLSModelNLS, x::AbstractVector)
    NLPModels.increment!(nls, :neval_obj)
    return ensure_res!(nls, x)
end

function NLPModels.grad!(nls::NLLSModelNLS, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(nls, :neval_grad)
    ensure_res!(nls, x)
    ensure_jac!(nls, x)
    mul!(g, transpose(nls.J_buf), nls.r_buf)
    return g
end

function NLPModels.objgrad!(nls::NLLSModelNLS, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(nls, :neval_obj)
    NLPModels.increment!(nls, :neval_grad)
    cost = ensure_res!(nls, x)
    ensure_jac!(nls, x)
    mul!(g, transpose(nls.J_buf), nls.r_buf)
    return cost, g
end

function NLPModels.reset!(nls::NLLSModelNLS)
    NLPModels.reset!(nls.counters)
    nls.res_valid = false
    nls.jac_valid = false
    fill!(nls.x_cached, NaN)
    return nls
end

function OptimizationVariableInterface.syncvars!(nls::NLLSModelNLS, x::AbstractVector)
    sync_flat_vars!(nls.problem, nls.layout, x)
    return nls.problem
end

OptimizationVariableInterface.syncvars!(::Model, nls::NLLSModelNLS, x::AbstractVector) =
    syncvars!(nls, x)

end # module NLPModelExt
