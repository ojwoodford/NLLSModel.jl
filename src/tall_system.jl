# Tall residual and Jacobian system assembly for NLLSModel

using Polyester

mutable struct NLSRobustStatus
    clamped_count::Int
    warned_clamp::Bool
end
NLSRobustStatus() = NLSRobustStatus(0, false)

struct TaskResult
    cost::Float64
    clamped::Int64
    err_code::Int64 # 0 = OK, 1 = ERR_DOMAIN
    err_val::Float64
end

struct NLSTallLayout
    K::Int                          # Total residual blocks
    M::Int                          # Total scalar residual length
    N::Int                          # Total unfixed variable degrees of freedom
    nnz::Int                        # Total structural nonzeros in sparse J
    resoffsets::Vector{Int}         # Length K + 1
    resblocksizes::Vector{Int}      # Length K
    varoffsets::Vector{Int}         # Length nblocks + 1
    varblocksizes::Vector{Int}      # Length nblocks
    blockindices::Vector{Int}       # Length length(problem.variables)
    nblocks::Int                    # Number of unfixed variable blocks
    costvec_start_idx::Vector{Int}  # Starting global cost index for each costvec
    dep_start_idx::Vector{Int}      # Starting dependency index in sparse_p for each cost
    sparse_p::Vector{Int}           # Base nzval index for each (cost, dep)
    col_strides::Vector{Int}        # Height H_b for each unfixed block
    is_dense::Bool
    linear_weights::Vector{Float64} # Precomputed sqrt(w) for linear kernels
    all_linear::Bool
    work::Int                       # Estimated work (see costwork), compared with min_work_per_thread
end

# Call `f(cost, k)` for every cost, with `k` its running index across all cost vectors. Recursing over the tuple of cost
# vectors gives `f` and the inner loop a concrete cost type each, so this stays type stable for any number of cost types.
@inline eachcost(f::F, costs) where F = _eachcost(f, values(costs), 1)
@inline _eachcost(f, ::Tuple{}, k) = nothing
@inline function _eachcost(f::F, vecs::Tuple, k) where F
    v = first(vecs)
    _eachcostvec(f, v, k)
    return _eachcost(f, Base.tail(vecs), k + length(v))
end
function _eachcostvec(f::F, v::AbstractVector, k) where F
    for i in eachindex(v)
        f(@inbounds(v[i]), k + i - 1)
    end
    return nothing
end

@inline has_unique_varindices(c) = (v = varindices(c); v isa Integer ? true : allunique(v))

@inline function cost_varflags(blockindices, c)
    vinds = varindices(c)
    if vinds isa Integer
        return (blockindices[vinds] != 0) ? 1 : 0
    else
        flags = 0 # An Int (not a StaticInt): the value is only known at run time
        @unroll for i in 1:MAX_ARGS
            if i <= length(vinds)
                b = @inbounds blockindices[vinds[i]]
                if b != 0
                    flags |= 1 << (i - 1)
                end
            end
        end
        return flags
    end
end

function build_tall_layout(problem::Model, unfixed, matrix_type::Symbol, max_dense_bytes::Int, robust::Symbol)
    # 1. Validate cost types and variable indices
    for costvec in values(problem.costs)
        if !(eltype(costvec) <: AbstractResidual)
            throw(ArgumentError("optimvars_nls only supports AbstractResidual costs; found $(eltype(costvec))"))
        end
    end
    eachcost(problem.costs) do c, k
        if !has_unique_varindices(c)
            throw(ArgumentError("Cost of type $(typeof(c)) has duplicate variable indices in varindices(c); unique variable indices per cost are required"))
        end
    end

    # 2. Sizing and layout arrays
    K = Int(countcosts(costnum, problem.costs))
    M = Int(countcosts(resnum, problem.costs))
    resoffsets = Vector{Int}(undef, K + 1)
    resblocksizes = Vector{Int}(undef, K)
    resoffsets[1] = 1
    dep_start_idx = Vector{Int}(undef, K + 1)
    dep_start_idx[1] = 1
    costvec_lengths = collect(map(length, values(problem.costs)))
    costvec_start_idx = Vector{Int}(undef, length(costvec_lengths))
    let k = 1
        for (i, n) in enumerate(costvec_lengths)
            costvec_start_idx[i] = k
            k += n
        end
    end

    linear_weights = Vector{Float64}(undef, K)
    warn_zero_weights = Int[]

    all_linear_ref = Ref(true)
    eachcost(problem.costs) do c, k
        m = Int(dynamic(nres(c)))
        resblocksizes[k] = m
        resoffsets[k + 1] = resoffsets[k] + m

        nd = Int(dynamic(ndeps(c)))
        dep_start_idx[k + 1] = dep_start_idx[k] + nd

        kernel = robustkernel(c)
        if c isa AbstractAdaptiveResidual
            all_linear_ref[] = false
            linear_weights[k] = 1.0
        elseif islinearkernel(kernel)
            w = linearweight(kernel)
            if w == 0.0
                push!(warn_zero_weights, k)
            end
            linear_weights[k] = sqrt(w)
        else
            all_linear_ref[] = false
            linear_weights[k] = 1.0
        end
    end
    all_linear = all_linear_ref[]

    if !isempty(warn_zero_weights)
        @warn "Cost(s) at index $(warn_zero_weights) have zero linear robust weight (w = 0.0); corresponding Jacobian rows are identically zero (rank-deficient)"
    end

    # 3. Variable blocks
    conv_unfixed = convertunfixed(unfixed, problem)
    nvars_total = length(problem.variables)
    blockindices = zeros(Int, nvars_total)
    nblocks = 0
    if conv_unfixed isa Integer
        nblocks = 1
        blockindices[conv_unfixed] = 1
    else
        for (i, unf) in enumerate(conv_unfixed)
            if unf
                nblocks += 1
                blockindices[i] = nblocks
            end
        end
    end

    # Check adaptive residuals for unfixed adaptive variables
    eachcost(problem.costs) do c, k
        if c isa AbstractAdaptiveResidual
            v1 = varindices(c)[1]
            if blockindices[v1] != 0
                throw(ArgumentError("optimvars_nls does not support unfixed AbstractAdaptiveRobustifier variables (found unfixed adaptive variable $v1 in cost $k)"))
            end
        end
    end

    varblocksizes = Vector{Int}(undef, nblocks)
    varoffsets = Vector{Int}(undef, nblocks + 1)
    varoffsets[1] = 1
    for i in 1:nvars_total
        b = blockindices[i]
        if b > 0
            varblocksizes[b] = Int(dynamic(nvars(problem.variables[i])))
        end
    end
    for b in 1:nblocks
        varoffsets[b + 1] = varoffsets[b] + varblocksizes[b]
    end
    N = nblocks > 0 ? (varoffsets[nblocks + 1] - 1) : 0

    # 4. Untouched variables check and column heights
    block_cost_counts = zeros(Int, nblocks)
    col_heights = zeros(Int, nblocks)
    eachcost(problem.costs) do c, k
        m = resblocksizes[k]
        vinds = varindices(c)
        nd = Int(dynamic(ndeps(c)))
        for d in 1:nd
            v = vinds isa Integer ? vinds : vinds[d]
            b = blockindices[v]
            if b > 0
                block_cost_counts[b] += 1
                col_heights[b] += m
            end
        end
    end

    empty_vars = [i for i in 1:nvars_total if blockindices[i] > 0 && block_cost_counts[blockindices[i]] == 0]
    if !isempty(empty_vars)
        @warn "Unfixed variable(s) $(empty_vars) are not referenced by any cost; corresponding Jacobian columns are identically zero (rank-deficient)"
    end

    # 5. Build sparse structure and sparse_p base pointers
    total_deps = dep_start_idx[K + 1] - 1
    sparse_p = zeros(Int, total_deps)
    current_col_offset = zeros(Int, nblocks)
    # Row indices of each block's columns, stored contiguously: block b occupies block_rows[block_row_start[b] + 1 : block_row_start[b] + col_heights[b]]
    block_row_start = zeros(Int, nblocks + 1)
    for b in 1:nblocks
        block_row_start[b + 1] = block_row_start[b] + col_heights[b]
    end
    block_rows = Vector{Int}(undef, block_row_start[nblocks + 1])
    eachcost(problem.costs) do c, k
        m = resblocksizes[k]
        vinds = varindices(c)
        nd = Int(dynamic(ndeps(c)))
        base_dep = dep_start_idx[k]
        for d in 1:nd
            v = vinds isa Integer ? vinds : vinds[d]
            b = blockindices[v]
            if b > 0
                sparse_p[base_dep + d - 1] = current_col_offset[b]
                dest = block_row_start[b] + current_col_offset[b]
                r_start = resoffsets[k]
                for r in 0:m-1
                    block_rows[dest + r + 1] = r_start + r
                end
                current_col_offset[b] += m
            end
        end
    end

    colptr = Vector{Int}(undef, N + 1)
    colptr[1] = 1
    nnz_total = 0
    for b in 1:nblocks
        nb = varblocksizes[b]
        hb = col_heights[b]
        for c in 1:nb
            col = varoffsets[b] + c - 1
            colptr[col + 1] = colptr[col] + hb
        end
        nnz_total += nb * hb
    end

    rowval = Vector{Int}(undef, nnz_total)
    for b in 1:nblocks
        hb = col_heights[b]
        if hb > 0
            for c in 0:varblocksizes[b]-1
                copyto!(rowval, colptr[varoffsets[b] + c], block_rows, block_row_start[b] + 1, hb)
            end
        end
    end

    J_sparse = SparseMatrixCSC{Float64, Int}(M, N, colptr, rowval, zeros(Float64, nnz_total))

    # Finalize sparse_p to absolute 1-based indices in J_csc.nzval
    eachcost(problem.costs) do c, k
        vinds = varindices(c)
        nd = Int(dynamic(ndeps(c)))
        base_dep = dep_start_idx[k]
        for d in 1:nd
            v = vinds isa Integer ? vinds : vinds[d]
            b = blockindices[v]
            if b > 0
                row_offset = sparse_p[base_dep + d - 1]
                sparse_p[base_dep + d - 1] = J_sparse.colptr[varoffsets[b]] + row_offset
            end
        end
    end

    # 6. Dense vs Sparse decision
    active_entries = sum(b -> varblocksizes[b] * col_heights[b], 1:nblocks; init=0)
    fill_fraction = active_entries / max(1, M * N)
    is_dense = false
    if matrix_type === :dense
        is_dense = true
    elseif matrix_type === :sparse
        is_dense = false
    else # :auto
        if (M * N * 8 <= max_dense_bytes) && (M * N <= 100_000 || fill_fraction >= 0.5 || (N <= 100 && fill_fraction >= 0.25))
            is_dense = true
        else
            is_dense = false
        end
    end

    J_proto = is_dense ? zeros(Float64, M, N) : J_sparse

    return NLSTallLayout(
        K, M, N, nnz_total,
        resoffsets, resblocksizes,
        varoffsets, varblocksizes,
        blockindices, nblocks,
        costvec_start_idx, dep_start_idx,
        sparse_p, col_heights,
        is_dense,
        linear_weights, all_linear,
        Int(countcosts(costwork, problem.costs))
    ), J_proto
end

@inline function dispatch_computeresjac(f, nvars_deps, varflags)
    maxflags = static(2 ^ dynamic(nvars_deps) - 1)
    if varflags == maxflags
        return f(maxflags)
    elseif dynamic(nvars_deps) <= 5
        return sirbsdispatch(static(0), maxflags - static(1), varflags, f)
    else
        return f(varflags)
    end
end

@inline is_nonfinite(x::Float64) = (reinterpret(UInt64, x) & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000
@inline is_nonfinite(x::Real) = is_nonfinite(Float64(x))
@inline is_finite(x) = !is_nonfinite(x)
@inline any_nonfinite(v) = any(is_nonfinite, v)

@inline safe_sqnorm(r_val::Number) = Float64(abs2(r_val))
@inline safe_sqnorm(r_val) = Float64(sum(abs2, r_val))

# Check non-finite status and write non-finite values into r without throwing
@inline function handle_nonfinite_res!(r, r_range, r_val, s)
    if any_nonfinite(r_val)
        r[r_range] .= r_val
    else
        r[r_range] .= NaN
    end
    return Inf
end

# Single residual evaluation returning (cost, clamped, err_code, err_val)
# How a non-linear robust kernel reweights a residual (shared by the residual and the residual+Jacobian evaluators)
const WEIGHT_ZERO = 0       # zero the residual (and Jacobian) rows
const WEIGHT_SCALE = 1      # scale rows by `scale` (sqrt of the kernel derivative)
const WEIGHT_CORRECTOR = 2  # second-order corrector: residual scale `scale / sqrt(1 + eps)`, plus a Jacobian projection
const WEIGHT_NAN = 3        # non-finite values: fill with NaN

# Returns (mode, cost, scale, eps, clamped count, error code, error value)
function robust_weights(kernel, s, robust::Symbol)
    if robust === :irls
        c_cost, dc, _ = robustifydcost(kernel, s)
        c_val = Float64(c_cost)
        dc_val = Float64(dc)
        if is_nonfinite(s) || is_nonfinite(c_val) || is_nonfinite(dc_val)
            return WEIGHT_NAN, (is_finite(c_val) ? NaN : 0.5 * c_val), 0.0, 0.0, 0, 0, 0.0
        end
        if dc_val > 0.0
            return WEIGHT_SCALE, 0.5 * c_val, sqrt(dc_val), 0.0, 0, 0, 0.0
        elseif dc_val == 0.0
            return WEIGHT_ZERO, 0.5 * c_val, 0.0, 0.0, 0, 0, 0.0
        end
        dc_tol = kernel_dc_tol(kernel)
        if dc_val < -dc_tol
            return WEIGHT_ZERO, 0.5 * c_val, 0.0, 0.0, 0, 1, dc_val
        end
        return WEIGHT_ZERO, 0.5 * c_val, 0.0, 0.0, 1, 0, 0.0
    elseif robust === :corrector
        c_cost, dc, d2c = robustifydcost(kernel, s)
        c_val = Float64(c_cost)
        dc_val = Float64(dc)
        d2c_val = Float64(d2c)
        if is_nonfinite(s) || is_nonfinite(c_val) || is_nonfinite(dc_val) || is_nonfinite(d2c_val)
            return WEIGHT_NAN, (is_finite(c_val) ? NaN : 0.5 * c_val), 0.0, 0.0, 0, 0, 0.0
        end
        if dc_val <= 0.0
            if dc_val == 0.0 && (s == 0.0 || d2c_val <= 0.0)
                return WEIGHT_ZERO, 0.5 * c_val, 0.0, 0.0, 0, 0, 0.0
            end
            dc_tol = kernel_dc_tol(kernel)
            if dc_val < -dc_tol || (s > 0.0 && d2c_val * s > dc_tol)
                return WEIGHT_ZERO, 0.5 * c_val, 0.0, 0.0, 0, 1, dc_val
            end
            return WEIGHT_ZERO, 0.5 * c_val, 0.0, 0.0, dc_val < 0.0 ? 1 : 0, 0, 0.0
        end
        if s == 0.0 || d2c_val == 0.0
            return WEIGHT_SCALE, 0.5 * c_val, sqrt(dc_val), 0.0, 0, 0, 0.0
        end
        raw_eps = (2.0 * d2c_val * s) / dc_val
        eps_i = raw_eps < -1.0/3.0 ? -1.0/3.0 : raw_eps
        if is_nonfinite(eps_i)
            return WEIGHT_NAN, NaN, 0.0, 0.0, 0, 0, 0.0
        end
        return WEIGHT_CORRECTOR, 0.5 * c_val, sqrt(dc_val), eps_i, 0, 0, 0.0
    end
    throw(ArgumentError("Unknown robust mode: $robust"))
end

function eval_single_res!(r, vars, cost, k::Int, layout::NLSTallLayout, robust::Symbol, status::NLSRobustStatus)
    vars_k = getvars(cost, vars)
    r_start = layout.resoffsets[k]
    m_k = layout.resblocksizes[k]
    r_range = r_start:(r_start + m_k - 1)

    if cost isa AbstractAdaptiveResidual
        kernel = vars_k[1]
        vars_for_res = vars_k[2:end]
    else
        kernel = robustkernel(cost)
        vars_for_res = vars_k
    end

    r_val = computeresidual(cost, vars_for_res...)
    s = safe_sqnorm(r_val)
    if is_nonfinite(s) || any_nonfinite(r_val)
        c_ret = handle_nonfinite_res!(r, r_range, r_val, s)
        return c_ret, 0, 0, 0.0
    end

    if robust === :none
        r[r_range] .= r_val
        return 0.5 * s, 0, 0, 0.0
    end

    if layout.all_linear || islinearkernel(kernel)
        w = layout.linear_weights[k]
        if w == 0.0
            r[r_range] .= 0.0
            return 0.0, 0, 0, 0.0
        else
            r[r_range] .= r_val .* w
            return 0.5 * (w * w) * s, 0, 0, 0.0
        end
    end

    # Non-linear robust path
    mode, c_ret, scale, eps_i, clamped, err, val = robust_weights(kernel, s, robust)
    if mode == WEIGHT_ZERO
        r[r_range] .= 0.0
    elseif mode == WEIGHT_NAN
        r[r_range] .= NaN
    elseif mode == WEIGHT_SCALE
        r[r_range] .= r_val .* scale
    else # WEIGHT_CORRECTOR
        r[r_range] .= r_val .* (scale / sqrt(1.0 + eps_i))
    end
    return c_ret, clamped, err, val
end

# Write zero into Jacobian columns/nzval for cost k
function zero_cost_jacobian!(J, k::Int, layout::NLSTallLayout, cost)
    nd = Int(dynamic(ndeps(cost)))
    vinds = varindices(cost)
    base_dep = layout.dep_start_idx[k]
    m_k = layout.resblocksizes[k]
    r_start = layout.resoffsets[k]

    for d in 1:nd
        v = vinds isa Integer ? vinds : vinds[d]
        b = layout.blockindices[v]
        if b > 0
            nb = layout.varblocksizes[b]
            if J isa SparseMatrixCSC
                p = layout.sparse_p[base_dep + d - 1]
                stride = layout.col_strides[b]
                for c_loc in 0:nb-1
                    col_p = p + c_loc * stride
                    for r_loc in 0:m_k-1
                        J.nzval[col_p + r_loc] = 0.0
                    end
                end
            else
                col_start = layout.varoffsets[b]
                for c_loc in 0:nb-1
                    col = col_start + c_loc
                    for r_loc in 0:m_k-1
                        J[r_start + r_loc, col] = 0.0
                    end
                end
            end
        end
    end
end

# Write non-finite into Jacobian columns/nzval for cost k
function nan_cost_jacobian!(J, k::Int, layout::NLSTallLayout, cost)
    nd = Int(dynamic(ndeps(cost)))
    vinds = varindices(cost)
    base_dep = layout.dep_start_idx[k]
    m_k = layout.resblocksizes[k]
    r_start = layout.resoffsets[k]

    for d in 1:nd
        v = vinds isa Integer ? vinds : vinds[d]
        b = layout.blockindices[v]
        if b > 0
            nb = layout.varblocksizes[b]
            if J isa SparseMatrixCSC
                p = layout.sparse_p[base_dep + d - 1]
                stride = layout.col_strides[b]
                for c_loc in 0:nb-1
                    col_p = p + c_loc * stride
                    for r_loc in 0:m_k-1
                        J.nzval[col_p + r_loc] = NaN
                    end
                end
            else
                col_start = layout.varoffsets[b]
                for c_loc in 0:nb-1
                    col = col_start + c_loc
                    for r_loc in 0:m_k-1
                        J[r_start + r_loc, col] = NaN
                    end
                end
            end
        end
    end
end

# Scatter scaled jacobian matrix into J
function scatter_cost_jacobian!(J, jac_scaled, k::Int, layout::NLSTallLayout, cost)
    nd = Int(dynamic(ndeps(cost)))
    vinds = varindices(cost)
    base_dep = layout.dep_start_idx[k]
    m_k = layout.resblocksizes[k]
    r_start = layout.resoffsets[k]
    loffset = 0

    for d in 1:nd
        v = vinds isa Integer ? vinds : vinds[d]
        b = layout.blockindices[v]
        if b > 0
            nb = layout.varblocksizes[b]
            if J isa SparseMatrixCSC
                p = layout.sparse_p[base_dep + d - 1]
                stride = layout.col_strides[b]
                for c_loc in 0:nb-1
                    col_p = p + c_loc * stride
                    for r_loc in 0:m_k-1
                        J.nzval[col_p + r_loc] = jac_scaled[r_loc + 1, loffset + c_loc + 1]
                    end
                end
            else
                col_start = layout.varoffsets[b]
                for c_loc in 0:nb-1
                    col = col_start + c_loc
                    for r_loc in 0:m_k-1
                        J[r_start + r_loc, col] = jac_scaled[r_loc + 1, loffset + c_loc + 1]
                    end
                end
            end
            loffset += nb
        end
    end
end

# Single residual and Jacobian evaluation returning (cost, clamped, err_code, err_val)
# Per-task scratch matrix for a scaled Jacobian (Triggs correction, robust weights), so that nothing is allocated per cost.
# The matrix is valid until the same task evaluates its next cost; `scatter_cost_jacobian!` consumes it straight away.
function tall_scratch(::Type{T}, m::Int, P::Int)::Matrix{T} where T
    cache = get!(() -> Dict{DataType, Any}(), task_local_storage(), :nllsmodel_tallcache)::Dict{DataType, Any}
    bysize = get!(() -> Dict{Tuple{Int, Int}, Matrix{T}}(), cache, T)::Dict{Tuple{Int, Int}, Matrix{T}}
    return get!(() -> Matrix{T}(undef, m, P), bysize, (m, P))
end

# `w * jac`: unrolled for static Jacobians, into the per-task scratch matrix for dynamic ones
@inline scaled_jacobian(jac::Union{StaticMatrix, Adjoint{<:Any, <:StaticVector}}, w) = jac .* w
function scaled_jacobian(jac::AbstractMatrix, w)
    out = tall_scratch(float(eltype(jac)), size(jac, 1), size(jac, 2))
    @inbounds for j in axes(jac, 2), i in axes(jac, 1)
        out[i, j] = jac[i, j] * w
    end
    return out
end

function eval_single_resjac!(r, J, vars, cost, k::Int, layout::NLSTallLayout, robust::Symbol, status::NLSRobustStatus)
    vars_k = getvars(cost, vars)
    r_start = layout.resoffsets[k]
    m_k = layout.resblocksizes[k]
    r_range = r_start:(r_start + m_k - 1)

    varflags = cost_varflags(layout.blockindices, cost)
    if varflags == 0
        # All variables are fixed
        c, cl, err, val = eval_single_res!(r, vars, cost, k, layout, robust, status)
        zero_cost_jacobian!(J, k, layout, cost)
        return c, cl, err, val
    end

    if cost isa AbstractAdaptiveResidual
        kernel = vars_k[1]
        vars_for_res = vars_k[2:end]
        varflags_res = varflags >> 1
        nvars_deps = ndeps(cost) - static(1)
    else
        kernel = robustkernel(cost)
        vars_for_res = vars_k
        varflags_res = varflags
        nvars_deps = ndeps(cost)
    end

    res_k, jac_k = dispatch_computeresjac(vflags -> computeresjac(vflags, cost, vars_for_res...), nvars_deps, varflags_res)
    s = safe_sqnorm(res_k)

    if is_nonfinite(s) || any_nonfinite(res_k) || any_nonfinite(jac_k)
        handle_nonfinite_res!(r, r_range, res_k, s)
        nan_cost_jacobian!(J, k, layout, cost)
        return Inf, 0, 0, 0.0
    end

    if robust === :none
        r[r_range] .= res_k
        scatter_cost_jacobian!(J, jac_k, k, layout, cost)
        return 0.5 * s, 0, 0, 0.0
    end

    if layout.all_linear || islinearkernel(kernel)
        w = layout.linear_weights[k]
        if w == 0.0
            r[r_range] .= 0.0
            zero_cost_jacobian!(J, k, layout, cost)
            return 0.0, 0, 0, 0.0
        else
            r[r_range] .= res_k .* w
            jac_scaled = scaled_jacobian(jac_k, w)
            scatter_cost_jacobian!(J, jac_scaled, k, layout, cost)
            return 0.5 * (w * w) * s, 0, 0, 0.0
        end
    end

    # Non-linear robust path
    mode, c_ret, scale, eps_i, clamped, err, val = robust_weights(kernel, s, robust)
    if mode == WEIGHT_ZERO
        r[r_range] .= 0.0
        zero_cost_jacobian!(J, k, layout, cost)
    elseif mode == WEIGHT_NAN
        r[r_range] .= NaN
        nan_cost_jacobian!(J, k, layout, cost)
    elseif mode == WEIGHT_SCALE
        r[r_range] .= res_k .* scale
        scatter_cost_jacobian!(J, scaled_jacobian(jac_k, scale), k, layout, cost)
    else # WEIGHT_CORRECTOR
        tau = sqrt(1.0 + eps_i)
        alpha = -eps_i / (1.0 + tau)
        inv_norm = inv(sqrt(s))
        sqrt_dc = scale
        r[r_range] .= res_k .* (sqrt_dc / tau)

        # Scale jacobian: jac_scaled = sqrt_dc * (jac_k - alpha * n * (n' * jac_k))
        if jac_k isa SMatrix
            n = res_k * inv_norm
            u = n' * jac_k
            jac_scaled = sqrt_dc * (jac_k - alpha * (n * u))
        else
            P = size(jac_k, 2)
            jac_scaled = tall_scratch(eltype(jac_k), m_k, P)
            for col_j in 1:P
                v_j = 0.0
                for row_r in 1:m_k
                    v_j += res_k[row_r] * jac_k[row_r, col_j]
                end
                v_j *= inv_norm
                for row_r in 1:m_k
                    n_r = res_k[row_r] * inv_norm
                    jac_scaled[row_r, col_j] = sqrt_dc * (jac_k[row_r, col_j] - alpha * n_r * v_j)
                end
            end
        end
        scatter_cost_jacobian!(J, jac_scaled, k, layout, cost)
    end
    return c_ret, clamped, err, val
end

# Check caller-supplied J matrix
function check_caller_jacobian!(J, layout::NLSTallLayout; prezeroed::Bool=false)
    if size(J) != (layout.M, layout.N)
        throw(DimensionMismatch("Jacobian matrix size $(size(J)) does not match expected size ($(layout.M), $(layout.N))"))
    end
    if J isa SparseMatrixCSC
        if length(J.nzval) != layout.nnz
            throw(DimensionMismatch("Sparse Jacobian nzval length $(length(J.nzval)) does not match expected $(layout.nnz)"))
        end
    else
        if !prezeroed
            fill!(J, 0.0)
        end
    end
end

@inline function sync_tall_vars!(problem::Model, unfixed_inds::Integer, vars)
    @inbounds problem.variables[unfixed_inds] = vars
    return problem.variables
end

@inline function sync_tall_vars!(problem::Model, unfixed_inds::Vector{Int}, vars)
    @inbounds for (k, idx) in enumerate(unfixed_inds)
        problem.variables[idx] = vars[k]
    end
    return problem.variables
end

@inline function sync_tall_vars!(problem::Model, ::Nothing, vars)
    if vars !== problem.variables
        copyto!(problem.variables, vars)
    end
    return problem.variables
end

# Callable structs for optimvars_nls
struct NLSResidual{P, L, U, NT, TR}
    problem::P
    layout::L
    unfixed_inds::U
    robust::Symbol
    status::NLSRobustStatus
    numthreads::NT
    min_work_per_thread::Int
    task_results::TR
end

struct NLSResJac{P, L, U, NT, TR}
    problem::P
    layout::L
    unfixed_inds::U
    robust::Symbol
    status::NLSRobustStatus
    numthreads::NT
    min_work_per_thread::Int
    task_results::TR
end

"""
    syncvars!(problem::Model, res!::Union{NLSResidual, NLSResJac}, x_opt) -> Model

Write the solution `x_opt` found for the evaluators returned by [`optimvars_nls`](@ref) back into `problem.variables`.
The evaluators copy every point they are evaluated at into the model, so after an optimizer returns the model holds the
last point *evaluated*, which may be a rejected trial. This sets the solution.
"""
function OptimizationVariableInterface.syncvars!(problem::Model, fn::Union{NLSResidual, NLSResJac}, x)
    sync_tall_vars!(problem, fn.unfixed_inds, x)
    return problem
end

clamped_count(fn::Union{NLSResidual, NLSResJac}) = fn.status.clamped_count
reset_clamped_count!(fn::Union{NLSResidual, NLSResJac}) = (fn.status.clamped_count = 0; fn.status.warned_clamp = false; fn)

clamped_count(flat::NamedTuple) = hasproperty(flat, :status) ? flat.status.clamped_count : clamped_count(flat.resjac!)
reset_clamped_count!(flat::NamedTuple) = (if hasproperty(flat, :status)
    flat.status.clamped_count = 0
    flat.status.warned_clamp = false
else
    reset_clamped_count!(flat.resjac!)
end; flat)

# Evaluate costs `rng` of one cost vector. Returns (cost, clamped count, first error code, its value)
@inline function eval_costvec_range(evalone, costvec, k_start::Int, rng)
    total = 0.0
    clamped = 0
    err_code = 0
    err_val = 0.0
    for i in rng
        c_k, cl, err, val = evalone(costvec[i], k_start + i - 1)
        total += c_k
        clamped += cl
        if err != 0 && err_code == 0
            err_code = err
            err_val = val
        end
    end
    return total, clamped, err_code, err_val
end

@inline combine_eval(a, b) = (a[1] + b[1], a[2] + b[2], a[3] != 0 ? a[3] : b[3], a[3] != 0 ? a[4] : b[4])

# Recurse over the tuple of cost vectors so each is processed with its concrete type (type stable for any number of cost types)
@inline eval_costvecs(evalone, ::Tuple{}, start_idx, nchunks, taskid, ind) = (0.0, 0, 0, 0.0)
@inline function eval_costvecs(evalone, costvecs::Tuple, start_idx, nchunks, taskid, ind)
    costvec = first(costvecs)
    head = eval_costvec_range(evalone, costvec, start_idx[ind], chunk_indices(length(costvec), nchunks, taskid))
    return combine_eval(head, eval_costvecs(evalone, Base.tail(costvecs), start_idx, nchunks, taskid, ind + 1))
end

function check_tall_status!(status, clamped_tot, err_code, err_val, fname)
    if err_code != 0
        throw(DomainError(err_val, "Tall least-squares reweighting requires non-negative kernel derivative dc >= 0"))
    end
    if clamped_tot > 0
        status.clamped_count += clamped_tot
        if !status.warned_clamp
            status.warned_clamp = true
            @warn "Slightly negative kernel derivative(s) were clamped to zero in tall least-squares evaluation (total clamped: $(status.clamped_count)). Query clamped_count($fname) to inspect."
        end
    end
    return nothing
end

# Serial or threaded evaluation of all costs with `evalone(cost, k)`, shared by the residual and residual+Jacobian evaluators
function tall_eval!(evalone::F, costvecs, layout, status, task_results, N_req::Int, min_work::Int, fname) where F
    if N_req <= 1 || layout.work < 2 * min_work
        tot_cost, clamped_tot, err_code, err_val = eval_costvecs(evalone, costvecs, layout.costvec_start_idx, 1, 1, 1)
        check_tall_status!(status, clamped_tot, err_code, err_val, fname)
        return tot_cost
    end
    N_chunks = min(N_req, layout.work ÷ min_work)
    for t in 1:N_chunks
        task_results[t] = TaskResult(0.0, 0, 0, 0.0)
    end
    Polyester.@batch for taskid in 1:N_chunks
        local_cost, local_clamped, local_err, local_val = eval_costvecs(evalone, costvecs, layout.costvec_start_idx, N_chunks, taskid, 1)
        task_results[taskid] = TaskResult(local_cost, local_clamped, local_err, local_val)
    end
    for t in 1:N_chunks
        if task_results[t].err_code != 0
            throw(DomainError(task_results[t].err_val, "Tall least-squares reweighting requires non-negative kernel derivative dc >= 0"))
        end
    end
    check_tall_status!(status, sum(t -> task_results[t].clamped, 1:N_chunks), 0, 0.0, fname)
    return sum(t -> task_results[t].cost, 1:N_chunks)
end

function (res!::NLSResidual)(r::AbstractVector{Float64}, vars)
    if length(r) != res!.layout.M
        throw(DimensionMismatch("Residual vector length $(length(r)) does not match expected $(res!.layout.M)"))
    end

    problem = res!.problem
    layout = res!.layout
    robust = res!.robust
    status = res!.status
    numthreads = res!.numthreads
    min_work = res!.min_work_per_thread

    full_vars = sync_tall_vars!(problem, res!.unfixed_inds, vars)

    return tall_eval!(
        (c, k) -> eval_single_res!(r, full_vars, c, k, layout, robust, status),
        values(problem.costs), layout, status, res!.task_results, Int(dynamic(numthreads)), min_work, "res!")
end

function (resjac!::NLSResJac)(r::AbstractVector{Float64}, J, vars; prezeroed::Bool=false)
    if length(r) != resjac!.layout.M
        throw(DimensionMismatch("Residual vector length $(length(r)) does not match expected $(resjac!.layout.M)"))
    end
    check_caller_jacobian!(J, resjac!.layout; prezeroed=prezeroed)

    problem = resjac!.problem
    layout = resjac!.layout
    robust = resjac!.robust
    status = resjac!.status
    numthreads = resjac!.numthreads
    min_work = resjac!.min_work_per_thread

    full_vars = sync_tall_vars!(problem, resjac!.unfixed_inds, vars)

    return tall_eval!(
        (c, k) -> eval_single_resjac!(r, J, full_vars, c, k, layout, robust, status),
        values(problem.costs), layout, status, resjac!.task_results, Int(dynamic(numthreads)), min_work, "resjac!")
end

"""
    optimvars_nls(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), min_work_per_thread=5000, robust=:none, matrix_type=:auto, max_dense_bytes=256*1024^2)

Constructs tall residual and Jacobian callable structs `(f, res!, resjac!, x0, J_proto)` for `problem`.
Unlike `optimvars`, `unfixed` and `numthreads` are keyword arguments.
"""
function optimvars_nls(
    problem::Model;
    unfixed = nothing,
    numthreads = StaticInt(Threads.nthreads()),
    min_work_per_thread::Int = 5000,
    robust::Symbol = :none,
    matrix_type::Symbol = :auto,
    max_dense_bytes::Int = 256 * 1024^2,
)
    if !(robust in (:none, :irls, :corrector))
        throw(ArgumentError("Unknown robust mode: $robust. Supported modes are :none, :irls, :corrector"))
    end

    if robust === :none
        for costvec in values(problem.costs)
            for c in costvec
                if c isa AbstractAdaptiveResidual || !(robustkernel(c) isa NoRobust)
                    throw(ArgumentError("optimvars_nls with robust=:none requires all costs to have NoRobust kernels; found $(typeof(c isa AbstractAdaptiveResidual ? c : robustkernel(c))). Pass robust=:irls or robust=:corrector instead."))
                end
            end
        end
    end

    conv_unfixed = convertunfixed(unfixed, problem)
    layout, J_proto = build_tall_layout(problem, conv_unfixed, matrix_type, max_dense_bytes, robust)

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
        f = SingleVarCostFunc(problem, unfixed_inds, static(numthreads), 2 * min_work_per_thread)
    else
        x0 = unfixed_inds === nothing ? problem.variables : problem.variables[unfixed_inds]
        f = MultiVarCostFunc(problem, unfixed_inds, static(numthreads), 2 * min_work_per_thread)
    end

    N_req = Int(dynamic(numthreads))
    task_results = [TaskResult(0.0, 0, 0, 0.0) for _ in 1:max(1, N_req)]
    status = NLSRobustStatus()

    res! = NLSResidual(problem, layout, unfixed_inds, robust, status, static(numthreads), min_work_per_thread, task_results)
    resjac! = NLSResJac(problem, layout, unfixed_inds, robust, status, static(numthreads), min_work_per_thread, task_results)

    return (f, res!, resjac!, x0, J_proto)
end

"""
    flat_optimvars_nls(problem::Model; unfixed=nothing, numthreads=static(1), min_work_per_thread=5000, robust=:none, matrix_type=:auto, max_dense_bytes=256*1024^2)

Constructs a flat Euclidean vector wrapper around `optimvars_nls`.
Returns a NamedTuple `(; f, res!, resjac!, jac!, x0, J_proto, layout, status)`.
Requires all unfixed variables to satisfy the `isadditive` trait.
"""
function flat_optimvars_nls(
    problem::Model;
    unfixed = nothing,
    numthreads = static(1),
    min_work_per_thread::Int = 5000,
    robust::Symbol = :none,
    matrix_type::Symbol = :auto,
    max_dense_bytes::Int = 256 * 1024^2,
)
    conv_unfixed = convertunfixed(unfixed, problem)

    # Validate that all unfixed variables satisfy the additive trait
    if conv_unfixed isa Integer
        check_additive_variable(problem.variables[conv_unfixed])
    else
        for (i, unf) in enumerate(conv_unfixed)
            if unf
                check_additive_variable(problem.variables[i])
            end
        end
    end

    layout = FlatVarLayout(problem, unfixed)
    f_orig, res!_orig, resjac!_orig, x0_struct, J_proto = optimvars_nls(
        problem;
        unfixed = unfixed,
        numthreads = numthreads,
        min_work_per_thread = min_work_per_thread,
        robust = robust,
        matrix_type = matrix_type,
        max_dense_bytes = max_dense_bytes,
    )

    x0 = Vector{Float64}(undef, layout.n)
    pack_vars!(x0, layout, x0_struct)

    M = res!_orig.layout.M
    r_scratch = Vector{Float64}(undef, M)

    @inline function flat_f(x::AbstractVector{Float64})
        unpacked = unpack_vars!(layout, x)
        return f_orig(unpacked)
    end

    @inline function flat_res!(r::AbstractVector{Float64}, x::AbstractVector{Float64})
        unpacked = unpack_vars!(layout, x)
        return res!_orig(r, unpacked)
    end

    @inline function flat_resjac!(r::AbstractVector{Float64}, J, x::AbstractVector{Float64}; prezeroed::Bool=false)
        if eltype(J) !== Float64
            throw(ArgumentError("J must have eltype Float64, got $(eltype(J))"))
        end
        unpacked = unpack_vars!(layout, x)
        return resjac!_orig(r, J, unpacked; prezeroed=prezeroed)
    end

    @inline function flat_jac!(J, x::AbstractVector{Float64}; prezeroed::Bool=false)
        return flat_resjac!(r_scratch, J, x; prezeroed=prezeroed)
    end

    return (;
        f = flat_f,
        res! = flat_res!,
        resjac! = flat_resjac!,
        jac! = flat_jac!,
        x0 = x0,
        J_proto = J_proto,
        layout = layout,
        status = resjac!_orig.status,
        r_scratch = r_scratch,
        layout_tall = res!_orig.layout,
    )
end

"""
    nls_callbacks(problem::Model; unfixed=nothing, numthreads=static(1), min_work_per_thread=5000, matrix_type=:auto, max_dense_bytes=256*1024^2)

Lightweight callback helper for `LeastSquaresOptim.jl`.
Accepts any non-adaptive `AbstractResidual` problem with linear kernels (`islinearkernel`).
Caches `r` and `cost` at `x_cached` while writing `J` directly into caller's matrix.
Returns `(; x0, r0, J0, nequ, nvar, f, res!, jac!, reset!)`.
"""
function nls_callbacks(
    problem::Model;
    unfixed = nothing,
    numthreads = static(1),
    min_work_per_thread::Int = 5000,
    matrix_type::Symbol = :auto,
    max_dense_bytes::Int = 256 * 1024^2,
)
    for costvec in values(problem.costs)
        for c in costvec
            if c isa AbstractAdaptiveResidual || !islinearkernel(robustkernel(c))
                throw(ArgumentError("nls_callbacks only supports linear kernels (NoRobust or finite non-negative Scaled linear kernels) on non-adaptive residuals; found $(typeof(c isa AbstractAdaptiveResidual ? c : robustkernel(c)))"))
            end
        end
    end

    robust_mode = all(c -> robustkernel(c) isa NoRobust, Iterators.flatten(values(problem.costs))) ? :none : :irls
    flat = flat_optimvars_nls(
        problem;
        unfixed = unfixed,
        numthreads = numthreads,
        min_work_per_thread = min_work_per_thread,
        robust = robust_mode,
        matrix_type = matrix_type,
        max_dense_bytes = max_dense_bytes,
    )

    N = flat.layout.n
    M = flat.layout_tall.M
    x_cached = fill(NaN, N)
    r_buf = zeros(Float64, M)
    last_cost = Ref(0.0)
    res_valid = Ref(false)

    x0 = copy(flat.x0)
    r0 = zeros(Float64, M)
    J0 = copy(flat.J_proto)

    function cb_res!(r::AbstractVector{Float64}, x::AbstractVector{Float64})
        if x != x_cached || !res_valid[]
            copyto!(x_cached, x)
            last_cost[] = flat.res!(r_buf, x)
            res_valid[] = true
        end
        copyto!(r, r_buf)
        return r
    end

    function cb_jac!(J, x::AbstractVector{Float64})
        if eltype(J) !== Float64
            throw(ArgumentError("J must have eltype Float64"))
        end
        last_cost[] = flat.resjac!(r_buf, J, x)
        copyto!(x_cached, x)
        res_valid[] = true
        return J
    end

    function cb_f(x::AbstractVector{Float64})
        if x == x_cached && res_valid[]
            return last_cost[]
        end
        return flat.f(x)
    end

    function cb_reset!()
        res_valid[] = false
        fill!(x_cached, NaN)
        last_cost[] = 0.0
        return nothing
    end

    return (;
        x0 = x0,
        r0 = r0,
        J0 = J0,
        nequ = M,
        nvar = N,
        f = cb_f,
        res! = cb_res!,
        jac! = cb_jac!,
        reset! = cb_reset!,
    )
end
