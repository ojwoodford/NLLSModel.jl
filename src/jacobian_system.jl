# Residual and Jacobian system assembly for NLLSModel

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

struct NLSTallLayout{S}
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
    work::Int                       # Jacobian-fill estimate, sum of m * (n + 1) (see `workestimates`), compared with min_work
    schur::S                        # With elimination, where each cost's rows go in the tables (`JacobianSchurMap`), otherwise nothing
end

# Elimination on the Jacobian path: each cost is a row block of the `SchurBlockJacobian`'s layout, which BlockLinearSystems
# decides; a cost writes its block for each unfixed variable through that variable's handle in its row block.
struct JacobianSchurMap
    structure::SchurJacobianStructure # Built once, at setup: the optimizer reuses a workspace only for the same object
    dep_target::Vector{Int}           # Per (cost, dependency), indexed as `sparse_p`: `blockentry(structure, cost, layout
                                      # block)`, or 0 for a fixed variable
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
        @unroll for i in 1:MAX_ARGS # COV_EXCL_LINE
            if i <= length(vinds)
                b = @inbounds blockindices[vinds[i]]
                if b != 0
                    flags |= 1 << (i - 1)
                end
            end
        end # COV_EXCL_LINE
        return flags
    end
end

# Total dimension of the unfixed variables (an index or a mask)
unfixeddims(problem::Model, unfixed::Integer) = Int(dynamic(nvars(problem.variables[unfixed])))
unfixeddims(problem::Model, unfixed) = sum(i -> Int(dynamic(nvars(problem.variables[i]))), findall(unfixed); init=0)

# 1. Validate cost types and variable indices. The first step of the layout, run by `optimvars_nls` before the partition,
# which needs valid costs.
function check_jacobian_costs(problem::Model)
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
    return nothing
end

# Elimination on the Jacobian path: the partition of the Hessian path (`schur_partition`), provided every eliminated variable is
# in a cost (QR cannot eliminate a block of the Jacobian with no rows). A single unfixed variable is never partitioned,
# as on the Hessian path.
function jacobian_schur_partition(problem::Model, unfixed, reduction::Symbol)
    (unfixed isa Integer || count(unfixed) == 1) && (check_single_reduction(reduction); return nothing)
    partition = schur_partition(problem, unfixed, reduction)
    isnothing(partition) && return nothing
    incost = falses(length(problem.variables))
    eachcost(problem.costs) do c, k
        vinds = varindices(c)
        for v in (vinds isa Integer ? (vinds,) : vinds)
            incost[v] = true
        end
    end
    for v in view(partition.unfixed_raw, partition.eliminated)
        if !incost[v]
            reduction === :eliminate && throw(ArgumentError("reduction = :eliminate on the Jacobian path needs every eliminated variable in a cost; variable $v is in none."))
            return nothing
        end
    end
    return partition
end

# The layout of the elimination, and a handle for each (cost, unfixed variable) to write through (see `JacobianSchurMap`)
function jacobian_schur_map(problem::Model, partition, resblocksizes::Vector{Int}, dep_start_idx::Vector{Int}, nchunks::Int)
    # BlockLinearSystems lays out the elimination from the description, with each cost's rows as a row block
    structure = SchurJacobianStructure(partition.blocksizes, partition.rowblocks, resblocksizes; eliminated=partition.eliminated, nchunks)
    layoutblock = zeros(Int, length(problem.variables)) # Layout block of each unfixed variable, 0 for a fixed one
    for (p, v) in enumerate(partition.unfixed_raw[blockorder(structure)])
        layoutblock[v] = p
    end
    K = length(resblocksizes)
    dep_target = zeros(Int, dep_start_idx[K + 1] - 1)
    eachcost(problem.costs) do c, k
        vinds = varindices(c)
        for d in 1:Int(dynamic(ndeps(c)))
            j = layoutblock[vinds isa Integer ? vinds : vinds[d]]
            dep_target[dep_start_idx[k] + d - 1] = j == 0 ? 0 : blockentry(structure, k, j)
        end
    end
    return JacobianSchurMap(structure, dep_target)
end

# The layout of the Jacobian system, and the Jacobian prototype: a `Matrix` or a `SparseMatrixCSC`, or with a `partition`
# (elimination) a `SchurBlockJacobian`, whose reduced storage `storage` chooses; its element type is `T`, the `precision`
# keyword (the residual stays Float64: a SchurBlockJacobian keeps it in a Float64 vector of its own)
function build_jacobian_layout(problem::Model, unfixed, storage::Symbol, max_dense_bytes::Int, partition=nothing, nchunks::Int=1, ::Type{T}=Float64) where T
    # 2. Sizing and layout arrays
    K = Int(countcosts(costnum, problem.costs))
    M = Int(countcosts(resnum, problem.costs))
    # A forced dense Jacobian above the cap is refused before anything is allocated (with elimination, `storage` is the
    # storage of the reduced system, checked below)
    isnothing(partition) && storage === :dense && check_densefits("Jacobian", M, unfixeddims(problem, unfixed), max_dense_bytes, "raise max_dense_bytes or use storage = :sparse", T)
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
    active_entries = sum(b -> varblocksizes[b] * col_heights[b], 1:nblocks; init=0)
    work = active_entries + M # sum over costs of m * n (the Jacobian entries) plus m (the residual)

    # With elimination: the tables, and where each cost goes in them. The storage of the reduced system follows the Hessian
    # path's rule (`reduced_storage`); the optimizer builds its own pattern for a sparse one.
    if !isnothing(partition)
        schur = jacobian_schur_map(problem, partition, resblocksizes, dep_start_idx, nchunks)
        is_dense, = reduced_storage(SchurStructure(schur.structure), storage, max_dense_bytes, "reduced system", T; needpattern=false)
        J_proto = SchurBlockJacobian{T, is_dense ? Matrix{T} : SparseMatrixCSC{T, Int}}(schur.structure)
        return NLSTallLayout(
            K, M, N, 0,
            resoffsets, resblocksizes,
            varoffsets, varblocksizes,
            blockindices, nblocks,
            costvec_start_idx, dep_start_idx,
            Int[], col_heights,
            false,
            linear_weights, all_linear,
            work, schur
        ), J_proto
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

    J_sparse = SparseMatrixCSC{T, Int}(M, N, colptr, rowval, zeros(T, nnz_total))

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
    fill_fraction = active_entries / max(1, M * N)
    is_dense = false
    if storage === :dense
        is_dense = true
    elseif storage === :sparse
        is_dense = false
    else # :auto
        if densefits(M, N, max_dense_bytes, T) && (M * N <= 100_000 || fill_fraction >= 0.5 || (N <= 100 && fill_fraction >= 0.25))
            is_dense = true
        else
            is_dense = false
        end
    end

    J_proto = is_dense ? zeros(T, M, N) : J_sparse

    return NLSTallLayout(
        K, M, N, nnz_total,
        resoffsets, resblocksizes,
        varoffsets, varblocksizes,
        blockindices, nblocks,
        costvec_start_idx, dep_start_idx,
        sparse_p, col_heights,
        is_dense,
        linear_weights, all_linear,
        work, nothing
    ), J_proto
end

@inline function dispatch_computeresjac(f, nvars_deps, varflags)
    maxflags = static(2 ^ dynamic(nvars_deps) - 1)
    if varflags == maxflags
        return f(maxflags)
    elseif dynamic(nvars_deps) <= 5
        return sirbsdispatch(static(0), maxflags - static(1), varflags, f)
    else
        # Fall back on dynamic dispatch, as the Hessian path does (gradhesshelper!)
        return f(static(varflags))
    end
end

@inline is_nonfinite(x::Float64) = (reinterpret(UInt64, x) & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000
@inline is_nonfinite(x::Real) = is_nonfinite(Float64(x))
@inline is_finite(x) = !is_nonfinite(x)
@inline any_nonfinite(v) = any(is_nonfinite, v)

@inline safe_sqnorm(r_val::Number) = Float64(abs2(r_val))
@inline safe_sqnorm(r_val) = Float64(sum(abs2, r_val))

# Check non-finite status and write non-finite values into a cost's residual rows `rv` without throwing
@inline function handle_nonfinite_res!(rv, r_val)
    if any_nonfinite(r_val)
        rv .= r_val
    else
        rv .= NaN
    end
    return Inf
end

# The rows of cost `k`'s residual: in `r`, or with elimination in its table of `J` (`r` is then not used)
@inline residual_rows(r, J, k::Int, layout::NLSTallLayout) =
    view(r, layout.resoffsets[k]:(layout.resoffsets[k] + layout.resblocksizes[k] - 1))
@inline residual_rows(r, J::SchurBlockJacobian, k::Int, layout::NLSTallLayout) = residual_block(J, k, layout.resblocksizes[k])

# Single residual evaluation returning (cost, clamped, err_code, err_val)
# How a non-linear robust kernel reweights a residual (shared by the residual and the residual+Jacobian evaluators)
const WEIGHT_ZERO = 0       # zero the residual (and Jacobian) rows
const WEIGHT_SCALE = 1      # scale rows by `scale` (sqrt of the kernel derivative)
const WEIGHT_CORRECTOR = 2  # second-order corrector: residual scale `scale / sqrt(1 + eps)`, plus a Jacobian projection
const WEIGHT_NAN = 3        # non-finite values: fill with NaN

# Returns (mode, cost, scale, eps, clamped count, error code, error value). The kernel's own derivatives decide: a zero second
# derivative (or s == 0) scales the rows (IRLS), a non-zero one adds the Triggs correction, as on the Hessian path.
function robust_weights(kernel, s)
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

# `rv` is the cost's residual rows (`residual_rows`)
function eval_single_res!(rv, vars, cost, k::Int, layout::NLSTallLayout, status::NLSRobustStatus)
    vars_k = getvars(cost, vars)

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
        c_ret = handle_nonfinite_res!(rv, r_val)
        return c_ret, 0, 0, 0.0
    end

    if layout.all_linear || islinearkernel(kernel)
        w = layout.linear_weights[k]
        if w == 0.0
            rv .= 0.0
            return 0.0, 0, 0, 0.0
        else
            rv .= r_val .* w
            return 0.5 * (w * w) * s, 0, 0, 0.0
        end
    end

    # Non-linear robust path
    mode, c_ret, scale, eps_i, clamped, err, val = robust_weights(kernel, s)
    if mode == WEIGHT_ZERO
        rv .= 0.0
    elseif mode == WEIGHT_NAN
        rv .= NaN
    elseif mode == WEIGHT_SCALE
        rv .= r_val .* scale
    else # WEIGHT_CORRECTOR
        rv .= r_val .* (scale / sqrt(1.0 + eps_i))
    end
    return c_ret, clamped, err, val
end

# Write cost `k`'s Jacobian into its blocks of `J`. `src` is the cost's (scaled) Jacobian, or one value (0.0 or NaN) for
# every entry. One loop over the cost's unfixed variables; one method of `write_block!` for each kind of `J`.
@inline jacobian_entry(src::Number, i, j) = src
@inline jacobian_entry(src, i, j) = src[i, j]
function write_cost_jacobian!(J, src, k::Int, layout::NLSTallLayout, cost)
    nd = Int(dynamic(ndeps(cost)))
    vinds = varindices(cost)
    dep = layout.dep_start_idx[k] - 1
    m_k = layout.resblocksizes[k]
    loffset = 0
    for d in 1:nd
        v = vinds isa Integer ? vinds : vinds[d]
        b = layout.blockindices[v]
        if b > 0
            nb = layout.varblocksizes[b]
            write_block!(J, src, loffset, k, dep + d, b, m_k, nb, layout)
            loffset += nb
        end
    end
    return nothing
end

# A sparse J: the block's stored entries, one column of the block every `col_strides[b]` entries of `nzval`
function write_block!(J::SparseMatrixCSC, src, loffset, k, dep, b, m, nb, layout::NLSTallLayout)
    p = layout.sparse_p[dep]
    stride = layout.col_strides[b]
    for c_loc in 0:nb-1
        col_p = p + c_loc * stride
        for r_loc in 0:m-1
            J.nzval[col_p + r_loc] = jacobian_entry(src, r_loc + 1, loffset + c_loc + 1)
        end
    end
end

# A dense J (a Matrix, or the optimizer's static copy of one): the cost's rows, the block's columns
function write_block!(J::AbstractMatrix, src, loffset, k, dep, b, m, nb, layout::NLSTallLayout)
    r_start = layout.resoffsets[k]
    col_start = layout.varoffsets[b]
    for c_loc in 0:nb-1
        col = col_start + c_loc
        for r_loc in 0:m-1
            J[r_start + r_loc, col] = jacobian_entry(src, r_loc + 1, loffset + c_loc + 1)
        end
    end
end

# Elimination: the cost's rows (its row block) in the columns of the variable's block
function write_block!(J::SchurBlockJacobian, src, loffset, k, dep, b, m, nb, layout::NLSTallLayout)
    dest = block(J, k, layout.schur.dep_target[dep], m, nb)
    for c_loc in 1:nb
        for r_loc in 1:m
            dest[r_loc, c_loc] = jacobian_entry(src, r_loc, loffset + c_loc)
        end
    end
end

# Single residual and Jacobian evaluation returning (cost, clamped, err_code, err_val)
# Per-task scratch matrix for a scaled Jacobian (Triggs correction, robust weights), so that nothing is allocated per cost.
# The matrix is valid until the same task evaluates its next cost; `write_cost_jacobian!` consumes it straight away.
function jacobian_scratch(::Type{T}, m::Int, P::Int)::Matrix{T} where T
    cache = get!(() -> Dict{DataType, Any}(), task_local_storage(), :nllsmodel_jacobiancache)::Dict{DataType, Any}
    bysize = get!(() -> Dict{Tuple{Int, Int}, Matrix{T}}(), cache, T)::Dict{Tuple{Int, Int}, Matrix{T}}
    return get!(() -> Matrix{T}(undef, m, P), bysize, (m, P))
end

# `w * jac`: unrolled for static Jacobians, into the per-task scratch matrix for dynamic ones
@inline scaled_jacobian(jac::Union{StaticMatrix, Adjoint{<:Any, <:StaticVector}}, w) = jac .* w
function scaled_jacobian(jac::AbstractMatrix, w)
    out = jacobian_scratch(float(eltype(jac)), size(jac, 1), size(jac, 2))
    @inbounds for j in axes(jac, 2), i in axes(jac, 1)
        out[i, j] = jac[i, j] * w
    end
    return out
end

# The kernel, the variables the residual takes, how many of them, and which are unfixed. For an adaptive residual the first
# variable is the kernel's own parameters.
@inline split_kernel(cost::AbstractAdaptiveResidual, vars_k, varflags) = (vars_k[1], Base.tail(vars_k), ndeps(cost) - static(1), varflags >> 1)
@inline split_kernel(cost, vars_k, varflags) = (robustkernel(cost), vars_k, ndeps(cost), varflags)

# The same kernel for every kind of `J`: only where the rows are written differs (`residual_rows`, `write_block!`)
function eval_single_resjac!(r, J, vars, cost, k::Int, layout::NLSTallLayout, status::NLSRobustStatus)
    vars_k = getvars(cost, vars)
    m_k = layout.resblocksizes[k]
    rv = residual_rows(r, J, k, layout)

    varflags = cost_varflags(layout.blockindices, cost)
    if varflags == 0
        # All variables are fixed: there are no Jacobian entries to write
        return eval_single_res!(rv, vars, cost, k, layout, status)
    end

    # Each of these is assigned once: a variable assigned in two branches and captured by the closure below is boxed
    # (`Core.Box`), which makes everything downstream inferred as `Any` and allocates on every call.
    kernel, vars_for_res, nvars_deps, varflags_res = split_kernel(cost, vars_k, varflags)

    res_k, jac_k = dispatch_computeresjac(vflags -> computeresjac(vflags, cost, vars_for_res...), nvars_deps, varflags_res)
    s = safe_sqnorm(res_k)

    if is_nonfinite(s) || any_nonfinite(res_k) || any_nonfinite(jac_k)
        handle_nonfinite_res!(rv, res_k)
        write_cost_jacobian!(J, NaN, k, layout, cost)
        return Inf, 0, 0, 0.0
    end

    if layout.all_linear || islinearkernel(kernel)
        w = layout.linear_weights[k]
        if w == 0.0
            rv .= 0.0
            write_cost_jacobian!(J, 0.0, k, layout, cost)
            return 0.0, 0, 0, 0.0
        else
            rv .= res_k .* w
            write_cost_jacobian!(J, scaled_jacobian(jac_k, w), k, layout, cost)
            return 0.5 * (w * w) * s, 0, 0, 0.0
        end
    end

    # Non-linear robust path
    mode, c_ret, scale, eps_i, clamped, err, val = robust_weights(kernel, s)
    if mode == WEIGHT_ZERO
        rv .= 0.0
        write_cost_jacobian!(J, 0.0, k, layout, cost)
    elseif mode == WEIGHT_NAN
        rv .= NaN
        write_cost_jacobian!(J, NaN, k, layout, cost)
    elseif mode == WEIGHT_SCALE
        rv .= res_k .* scale
        write_cost_jacobian!(J, scaled_jacobian(jac_k, scale), k, layout, cost)
    else # WEIGHT_CORRECTOR
        tau = sqrt(1.0 + eps_i)
        alpha = -eps_i / (1.0 + tau)
        inv_norm = inv(sqrt(s))
        sqrt_dc = scale
        rv .= res_k .* (sqrt_dc / tau)

        # Scale jacobian: jac_scaled = sqrt_dc * (jac_k - alpha * n * (n' * jac_k))
        if jac_k isa SMatrix
            n = res_k * inv_norm
            u = n' * jac_k
            jac_scaled = sqrt_dc * (jac_k - alpha * (n * u))
        else
            P = size(jac_k, 2)
            jac_scaled = jacobian_scratch(eltype(jac_k), m_k, P)
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
        write_cost_jacobian!(J, jac_scaled, k, layout, cost)
    end
    return c_ret, clamped, err, val
end

# Check caller-supplied J matrix. With `prezeroed = false` (the default) a dense `J` is filled with zeros first, so it may hold
# anything. With `prezeroed = true` it is not touched here: the structural zeros of `J` are taken to be in place already. That is
# valid for a `J` that started as zeros (`optimvars_nls` returns such a prototype) and has only ever been written by an evaluator
# of this model, because every call writes every entry that can be non-zero (a cost's whole block, with explicit zeros for a zero
# weight and NaN for a non-finite value), and no other entry.
# With elimination the zeros of the tables are the entries no cost writes, and the damping rows, which only the optimizer's
# copy of the tables uses. The tables must be those of this evaluator's structure, which says where each cost goes.
function check_caller_jacobian!(J::SchurBlockJacobian, layout::NLSTallLayout; prezeroed::Bool=false)
    (layout.schur isa JacobianSchurMap && J.structure === layout.schur.structure) ||
        throw(ArgumentError("This SchurBlockJacobian has another structure: pass the prototype that optimvars_nls returned for this model, or a copy_structure of it."))
    prezeroed || zero!(J)
    return nothing
end
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

@inline function sync_jacobian_vars!(problem::Model, unfixed_inds::Integer, vars)
    @inbounds problem.variables[unfixed_inds] = vars
    return problem.variables
end

@inline function sync_jacobian_vars!(problem::Model, unfixed_inds::Vector{Int}, vars)
    @inbounds for (k, idx) in enumerate(unfixed_inds)
        problem.variables[idx] = vars[k]
    end
    return problem.variables
end

# Every variable unfixed: `vars` is a complete set of variables and is evaluated as it is. It is not copied into the model:
# `vars` may be the optimizer's current point, which is `problem.variables` itself, and a trial must not overwrite it.
# (The same rule as `sync_unfixed_vars!` on the Hessian path.)
@inline sync_jacobian_vars!(::Model, ::Nothing, vars) = vars

# The residual vector: one entry per residual row, except with elimination, where the residuals are written into the
# tables of `J` and `r` is not used (the optimizer passes an empty one)
check_caller_residual(r, J, layout::NLSTallLayout) = length(r) == layout.M ||
    throw(DimensionMismatch("Residual vector length $(length(r)) does not match expected $(layout.M)"))
check_caller_residual(r, ::SchurBlockJacobian, layout::NLSTallLayout) = nothing

# Callable structs for optimvars_nls
struct NLSResidual{P, L, U, TR}
    problem::P
    layout::L
    unfixed_inds::U
    status::NLSRobustStatus
    min_work::Int # Per-thread floor
    nthreads::Int # Threads the fill uses, `threadcount(layout.work, min_work, numthreads, costsplit)`
    task_results::TR
end

struct NLSResJac{P, L, U, TR}
    problem::P
    layout::L
    unfixed_inds::U
    status::NLSRobustStatus
    min_work::Int # Per-thread floor
    nthreads::Int # Threads the fill uses, `threadcount(layout.work, min_work, numthreads, costsplit)`
    task_results::TR
end

"""
    syncvars!(problem::Model, res!::Union{NLSResidual, NLSResJac}, x_opt) -> Model

Write the solution `x_opt` found for the evaluators returned by [`optimvars_nls`](@ref) back into `problem.variables`.
With a subset or a single variable unfixed, the evaluators copy every point they evaluate into the unfixed slots of the model,
so after an optimizer returns the model may hold the last point *evaluated*, which may be a rejected trial. With every variable
unfixed they evaluate the point given and copy nothing, so the model changes only when the optimizer updates `x0` in place.
Either way this sets the solution.
"""
function OptimizationVariableInterface.syncvars!(problem::Model, fn::Union{NLSResidual, NLSResJac}, x)
    if isnothing(fn.unfixed_inds)
        x === problem.variables || copyto!(problem.variables, x)
    else
        sync_jacobian_vars!(problem, fn.unfixed_inds, x)
    end
    return problem
end

"""
    islinear(fn) -> Bool

For the evaluators returned by [`optimvars_nls`](@ref) (`res!` or `resjac!`) or the named tuple returned by
[`flat_optimvars_nls`](@ref): `true` when every cost is a non-adaptive residual with a linear kernel (`NoRobust`, or `Scaled` with a finite non-negative height).
Then `r` is the true (constant-scaled) residual and the cost returned is exactly `½‖r‖²`, which solvers that derive the cost from
`r` rely on. Otherwise some rows are reweighted and the cost is the robust cost, not `½‖r‖²`.
"""
islinear(fn::Union{NLSResidual, NLSResJac}) = fn.layout.all_linear
islinear(flat::NamedTuple) = flat.layout_jacobian.all_linear

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

function check_jacobian_status!(status, clamped_tot, err_code, err_val, fname)
    if err_code != 0
        throw(DomainError(err_val, "Least-squares reweighting requires non-negative kernel derivative dc >= 0"))
    end
    if clamped_tot > 0
        status.clamped_count += clamped_tot
        if !status.warned_clamp
            status.warned_clamp = true
            @warn "Slightly negative kernel derivative(s) were clamped to zero in least-squares evaluation on the Jacobian path (total clamped: $(status.clamped_count)). Query clamped_count($fname) to inspect."
        end
    end
    return nothing
end

# Serial or threaded evaluation of all costs with `evalone(cost, k)`, shared by the residual and residual+Jacobian evaluators
function jacobian_eval!(evalone::F, costvecs, layout, status, task_results, N_chunks::Int, fname) where F
    if N_chunks <= 1
        tot_cost, clamped_tot, err_code, err_val = eval_costvecs(evalone, costvecs, layout.costvec_start_idx, 1, 1, 1)
        check_jacobian_status!(status, clamped_tot, err_code, err_val, fname)
        return tot_cost
    end
    for t in 1:N_chunks
        task_results[t] = TaskResult(0.0, 0, 0, 0.0)
    end
    args = threadargs((evalone, costvecs, layout.costvec_start_idx, task_results))
    GC.@preserve evalone costvecs layout task_results begin
        @batch for taskid in 1:N_chunks
            evalone_, costvecs_, start_idx_, task_results_ = fromthreadargs(args)
            local_cost, local_clamped, local_err, local_val = eval_costvecs(evalone_, costvecs_, start_idx_, N_chunks, taskid, 1)
            task_results_[taskid] = TaskResult(local_cost, local_clamped, local_err, local_val)
        end # COV_EXCL_LINE
    end
    for t in 1:N_chunks
        if task_results[t].err_code != 0
            throw(DomainError(task_results[t].err_val, "Least-squares reweighting requires non-negative kernel derivative dc >= 0"))
        end
    end
    check_jacobian_status!(status, sum(t -> task_results[t].clamped, 1:N_chunks), 0, 0.0, fname)
    return sum(t -> task_results[t].cost, 1:N_chunks)
end

function (res!::NLSResidual)(r::AbstractVector{Float64}, vars)
    if length(r) != res!.layout.M
        throw(DimensionMismatch("Residual vector length $(length(r)) does not match expected $(res!.layout.M)"))
    end

    problem = res!.problem
    layout = res!.layout
    status = res!.status

    full_vars = sync_jacobian_vars!(problem, res!.unfixed_inds, vars)

    return jacobian_eval!(
        (c, k) -> eval_single_res!(residual_rows(r, nothing, k, layout), full_vars, c, k, layout, status),
        values(problem.costs), layout, status, res!.task_results, res!.nthreads, "res!")
end

# `prezeroed = true` says the structural zeros of the dense `J` are already in place (see `check_caller_jacobian!`): the optimizer
# hands over a `J` that started as zeros and that only this evaluator writes. The default fills `J` with zeros first.
function (resjac!::NLSResJac)(r::AbstractVector{Float64}, J, vars; prezeroed::Bool=false)
    check_caller_residual(r, J, resjac!.layout)
    check_caller_jacobian!(J, resjac!.layout; prezeroed=prezeroed)

    problem = resjac!.problem
    layout = resjac!.layout
    status = resjac!.status

    full_vars = sync_jacobian_vars!(problem, resjac!.unfixed_inds, vars)

    return jacobian_eval!(
        (c, k) -> eval_single_resjac!(r, J, full_vars, c, k, layout, status),
        values(problem.costs), layout, status, resjac!.task_results, resjac!.nthreads, "resjac!")
end

"""
    optimvars_nls(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work=$MIN_WORK_PER_THREAD_JACOBIAN, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64)

Constructs the residual and Jacobian callable structs `(f, res!, resjac!, x0, J_proto)` for `problem`.

There is no robust mode: each cost's kernel decides how its rows enter, as on the Hessian path.

| What the kernel returns from `robustifydcost` | Rows of that cost |
|:---|:---|
| a linear kernel (`NoRobust`, or `Scaled` with a finite non-negative height) | unweighted or scaled by a constant |
| second derivative zero (e.g. `HuberKernel`) | IRLS: scaled by the square root of the first derivative |
| second derivative non-zero (e.g. `Huber2oKernel`) | IRLS with the Triggs correction |

So choosing first- or second-order treatment means choosing the kernel type. `resjac!(r, J, x)` returns the true cost
`cost(model)` at `x`; this equals `½‖r‖²` only when [`islinear`](@ref)`(resjac!)` is true.

`J_proto` is a `Matrix{Float64}` or a `SparseMatrixCSC{Float64, Int}`, chosen by `storage`. In place of a dense prototype,
`resjac!` takes any dense `AbstractMatrix` of its size (SecondOrderOptimization hands it an `MMatrix` copy, and an
`MVector` for `r`, when it solves a small single static variable with its static system). With elimination (`reduction`) it
is a BlockLinearSystems `SchurBlockJacobian{Float64, RType}`: one table per eliminated variable, holding the rows of the
costs that involve it, and tables for the other costs. `resjac!(r, J, x)` then writes each cost's residual and Jacobian
rows into its table, and leaves `r` alone (pass any vector, for example an empty one); `res!(r, x)` still writes the
whole residual into `r`. `RType`, `Matrix{Float64}` or `SparseMatrixCSC{Float64, Int}`, is the storage of the reduced
system, chosen by `storage` and `max_dense_bytes` by the Hessian path's rule. `x0` then holds the unfixed variables
in the partition's order, retained first; [`syncvars!`](@ref) writes a solution back.

What `optimize!(model, options; formulation = :jacobian)` does with each prototype:

$JACOBIAN_FORMULATION_DOC

$KEYWORDS_DOC"""
function optimvars_nls(
    problem::Model;
    unfixed = nothing,
    numthreads = StaticInt(Threads.nthreads()),
    reduction::Symbol = :auto,
    storage::Symbol = :auto,
    max_dense_bytes::Integer = MAX_DENSE_BYTES,
    min_work::Integer = MIN_WORK_PER_THREAD_JACOBIAN,
    min_work_cost::Integer = MIN_WORK_PER_THREAD_COST,
    precision::Type = Float64,
)
    check_reduction(reduction)
    check_storage(storage)
    check_precision(precision)
    check_jacobian_costs(problem)
    conv_unfixed = convertunfixed(unfixed, problem)
    partition = jacobian_schur_partition(problem, conv_unfixed, reduction)
    layout, J_proto = build_jacobian_layout(problem, conv_unfixed, storage, Int(max_dense_bytes), partition, Int(dynamic(numthreads)), precision)

    if !isnothing(partition)
        # The variables in the layout's order (retained, then eliminated), which is the column order of the tables
        is_single = false
        unfixed_inds = partition.unfixed_raw[blockorder(layout.schur.structure)]
    elseif conv_unfixed isa Integer
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
        f = SingleVarCostFunc(problem, unfixed_inds, static(numthreads), min_work_cost)
    else
        x0 = isnothing(unfixed_inds) ? problem.variables : problem.variables[unfixed_inds]
        f = MultiVarCostFunc(problem, unfixed_inds, static(numthreads), min_work_cost)
    end

    nthreads = threadcount(layout.work, min_work, dynamic(numthreads), costsplit(problem.costs))
    task_results = [TaskResult(0.0, 0, 0, 0.0) for _ in 1:nthreads]
    status = NLSRobustStatus()

    res! = NLSResidual(problem, layout, unfixed_inds, status, Int(min_work), nthreads, task_results)
    resjac! = NLSResJac(problem, layout, unfixed_inds, status, Int(min_work), nthreads, task_results)

    return (f, res!, resjac!, x0, J_proto)
end

"""
    flat_optimvars_nls(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work=$MIN_WORK_PER_THREAD_JACOBIAN, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64)

Constructs a flat Euclidean vector wrapper around `optimvars_nls`.
Returns a NamedTuple `(; f, res!, resjac!, jac!, x0, J_proto, layout, status)`. Its Jacobian is a plain matrix, for
external solvers, so it never eliminates: `reduction = :auto` does not eliminate here, and `:eliminate` is an `ArgumentError`.
Accepts any unfixed variable whose update is plain addition, `update(v, δ) ≈ v + δ`, and that packs into a flat
`Float64` vector: a number, or an `AbstractVector` of numbers that can be built from a vector of its values (`Float64`,
`SVector{N, Float64}`, `Vector{Float64}`, or a type of your own). Nothing needs declaring: each variable type is checked
numerically, once, here. A type that cannot be packed, or whose update is not addition, is an `ArgumentError` saying which.

$KEYWORDS_DOC"""
function flat_optimvars_nls(
    problem::Model;
    unfixed = nothing,
    numthreads = StaticInt(Threads.nthreads()),
    reduction::Symbol = :auto,
    storage::Symbol = :auto,
    max_dense_bytes::Integer = MAX_DENSE_BYTES,
    min_work::Integer = MIN_WORK_PER_THREAD_JACOBIAN,
    min_work_cost::Integer = MIN_WORK_PER_THREAD_COST,
    precision::Type = Float64,
)
    reduction = check_plain_reduction(reduction, "flat_optimvars_nls")
    check_plain_precision(precision, "flat_optimvars_nls")
    conv_unfixed = convertunfixed(unfixed, problem)

    check_flat_variables(problem, conv_unfixed)

    layout = FlatVarLayout(problem, unfixed)
    f_orig, res!_orig, resjac!_orig, x0_struct, J_proto = optimvars_nls(
        problem;
        unfixed = unfixed,
        numthreads = numthreads,
        reduction = reduction,
        storage = storage,
        max_dense_bytes = max_dense_bytes,
        min_work = min_work,
        min_work_cost = min_work_cost,
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
        layout_jacobian = res!_orig.layout,
    )
end

# The error of the linear-kernel-only paths, naming the first offending cost
function throw_nonlinear(name, problem)
    for costvec in values(problem.costs)
        for c in costvec
            if c isa AbstractAdaptiveResidual || !islinearkernel(robustkernel(c))
                throw(ArgumentError("$name only supports linear kernels (NoRobust or finite non-negative Scaled linear kernels) on non-adaptive residuals; found $(typeof(c isa AbstractAdaptiveResidual ? c : robustkernel(c)))"))
            end
        end
    end
    throw(ArgumentError("$name only supports linear kernels (NoRobust or finite non-negative Scaled linear kernels) on non-adaptive residuals"))
end

"""
    nls_callbacks(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work=$MIN_WORK_PER_THREAD_JACOBIAN, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64)

Lightweight callback helper for `LeastSquaresOptim.jl`.
Accepts any non-adaptive `AbstractResidual` problem in which every cost has a linear kernel (`NoRobust`, or `Scaled` with a finite non-negative height).
Caches `r` and `cost` at `x_cached` while writing `J` directly into caller's matrix.
Returns `(; x0, r0, J0, nequ, nvar, f, res!, jac!, reset!)`.
"""
function nls_callbacks(
    problem::Model;
    unfixed = nothing,
    numthreads = StaticInt(Threads.nthreads()),
    reduction::Symbol = :auto,
    storage::Symbol = :auto,
    max_dense_bytes::Integer = MAX_DENSE_BYTES,
    min_work::Integer = MIN_WORK_PER_THREAD_JACOBIAN,
    min_work_cost::Integer = MIN_WORK_PER_THREAD_COST,
    precision::Type = Float64,
)
    check_plain_reduction(reduction, "nls_callbacks")
    check_plain_precision(precision, "nls_callbacks")
    flat = flat_optimvars_nls(
        problem;
        unfixed = unfixed,
        numthreads = numthreads,
        reduction = reduction,
        storage = storage,
        max_dense_bytes = max_dense_bytes,
        min_work = min_work,
        min_work_cost = min_work_cost,
    )

    islinear(flat) || throw_nonlinear("nls_callbacks", problem)

    N = flat.layout.n
    M = flat.layout_jacobian.M
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
