using Static, StaticArrays, SparseArrays, LinearAlgebra

# Conversions for unfixed
convertunfixed(::Nothing, problem) = trues(length(problem.variables))
convertunfixed(unfixed::Type, problem) = BitVector(isa(v, unfixed) for v in problem.variables)
convertunfixed(unfixed::Integer, problem) = unfixed
convertunfixed(unfixed::AbstractVector{Bool}, problem) = BitVector(unfixed)
function convertunfixed(unfixed::AbstractVector{<:Integer}, problem)
    mask = falses(length(problem.variables))
    mask[unfixed] .= true
    return mask
end
convertunfixed(unfixed, problem) = unfixed

# Single-variable objective and gradient/Hessian functors
struct SingleVarCostFunc{P, U}
    problem::P
    unfixed::U
    work::Int     # Cost-evaluation estimate (see `workestimates`), computed at setup
    min_work::Int # Per-thread floor
    nthreads::Int # Threads the cost is evaluated on, `threadcount(work, min_work, numthreads, costsplit)`
end
SingleVarCostFunc(problem, unfixed, numthreads, min_work::Integer=MIN_WORK_PER_THREAD_COST) =
    SingleVarCostFunc(problem, unfixed, costestimate(problem.costs), Int(min_work), threadcount(costestimate(problem.costs), min_work, dynamic(numthreads), costsplit(problem.costs)))
@inline function (f::SingleVarCostFunc)(x)
    @inbounds f.problem.variables[f.unfixed] = x
    return cost_multithreaded(f.problem.variables, f.problem.costs, f.nthreads)
end

struct SingleVarCostGradHess{LS, P, U}
    linsystem::LS
    problem::P
    unfixed::U
end
@inline makels(::UniVariateLSstatic_ls{N, N2}, H, g) where {N, N2} = UniVariateLSstatic_ls{N, N2}(H, g)
@inline makels(::UniVariateLSdynamic_ls, H, g) = UniVariateLSdynamic_ls(H, g)

@inline function (f::SingleVarCostGradHess)(g, H, x)
    @inbounds f.problem.variables[f.unfixed] = x
    if f.linsystem isa LinearSystemShared
        if H isa typeof(f.linsystem.ls[1].A) && g isa typeof(f.linsystem.ls[1].b)
            ls1 = makels(f.linsystem.ls[1], H, g)
            target = replace_first_ls(f.linsystem, ls1)
            return costgradhess!(target, f.problem.variables, f.problem.costs)
        else
            c = costgradhess!(f.linsystem, f.problem.variables, f.problem.costs)
            copyto!(g, getb(f.linsystem))
            copyto!(H, getA(f.linsystem))
            return c
        end
    else
        if H isa typeof(f.linsystem.ls.A) && g isa typeof(f.linsystem.ls.b)
            target_ls = LinearSystem(f.linsystem.x, makels(f.linsystem.ls, H, g))
            zero!(target_ls)
            return costgradhess_nozero!(target_ls, f.problem.variables, f.problem.costs)
        else
            c = costgradhess!(f.linsystem, f.problem.variables, f.problem.costs)
            copyto!(g, getb(f.linsystem))
            copyto!(H, getA(f.linsystem))
            return c
        end
    end
end

# Multi-variable objective and gradient/Hessian functors
@inline sync_unfixed_vars!(::Model, ::Nothing, vars) = vars
@inline function sync_unfixed_vars!(problem::Model, unfixed_inds::Vector{Int}, vars)
    @inbounds for (k, idx) in enumerate(unfixed_inds)
        problem.variables[idx] = vars[k]
    end
    return problem.variables
end

struct MultiVarCostFunc{P, U}
    problem::P
    unfixed_inds::U
    work::Int     # Cost-evaluation estimate (see `workestimates`), computed at setup
    min_work::Int # Per-thread floor
    nthreads::Int # Threads the cost is evaluated on, `threadcount(work, min_work, numthreads, costsplit)`
end
MultiVarCostFunc(problem, unfixed_inds, numthreads, min_work::Integer=MIN_WORK_PER_THREAD_COST) =
    MultiVarCostFunc(problem, unfixed_inds, costestimate(problem.costs), Int(min_work), threadcount(costestimate(problem.costs), min_work, dynamic(numthreads), costsplit(problem.costs)))
@inline function (f::MultiVarCostFunc)(vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    return cost_multithreaded(full_vars, f.problem.costs, f.nthreads)
end

# On several threads `linsystem` is a `LinearSystemShared`: one copy of the system per thread, summed afterwards, so each
# cost is linearized once, by one thread
struct MultiVarDenseFDF{LS, P, U}
    linsystem::LS
    problem::P
    unfixed_inds::U
end
@inline function (f::MultiVarDenseFDF)(g, H_mat::Matrix{Float64}, vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    orig_A = getA(f.linsystem)
    bdm = DenseBlockLinearSystem{Float64, Matrix{Float64}}(H_mat, orig_A.rowblockoffsets, orig_A.columnblockoffsets, orig_A.rowblocksizes, orig_A.columnblocksizes)
    if f.linsystem isa LinearSystemShared
        target = replace_first_ls(f.linsystem, MultiVariateLSdense_ls(bdm, g))
        c = costgradhess!(target, full_vars, f.problem.costs)
    else
        target_ls = LinearSystem(f.linsystem.x, MultiVariateLSdense_ls(bdm, g))
        zero!(target_ls)
        c = costgradhess_nozero!(target_ls, full_vars, f.problem.costs)
    end
    symmetrifyfull!(bdm)
    return c
end

struct MultiVarSparseFDF{LS, P, U}
    linsystem::LS
    problem::P
    unfixed_inds::U
end
@inline function (f::MultiVarSparseFDF)(g, H_csc::SparseMatrixCSC{Float64, Int}, vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    if f.linsystem isa LinearSystemShared
        target = replace_first_ls(f.linsystem, MultiVariateLSsparse_ls(f.linsystem.ls[1].A, g))
        c = costgradhess!(target, full_vars, f.problem.costs)
    else
        target_ls = LinearSystem(f.linsystem.x, MultiVariateLSsparse_ls(f.linsystem.ls.A, g))
        zero!(target_ls)
        c = costgradhess_nozero!(target_ls, full_vars, f.problem.costs)
    end
    copyto!(H_csc, getA(f.linsystem), f.linsystem.x.sparseindices)
    return c
end

struct MultiVarSchurFDF{LS, P, U, PL}
    linsystem::LS
    problem::P
    unfixed_inds::U
    plan::PL # SchurThreadPlan when linearizing on several threads, otherwise nothing
end
@inline function (f::MultiVarSchurFDF)(g, H_schur::SchurBlockMatrix, vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    if f.linsystem isa LinearSystemShared
        c = schur_threaded!(f.linsystem, f.plan, H_schur, g, full_vars, f.problem.costs)
    else
        target_ls = LinearSystem(f.linsystem.x, MultiVariateLSSchur_ls(H_schur, g))
        zero!(target_ls)
        c = costgradhess_nozero!(target_ls, full_vars, f.problem.costs)
    end
    return c
end

"""
    syncvars!(problem::Model, fdf!, x_opt) -> Model

Write the solution `x_opt` found for the functors returned by [`optimvars`](@ref) back into `problem.variables`.
`fdf!` (or `f`) knows which problem variables `x_opt` holds, and in what order (variables are sorted by dimension
into blocks when they are not already grouped), so this is the supported way to set the final solution. After
`optimize` the problem otherwise holds the last point *evaluated*, which may be a rejected step.
"""
function OptimizationVariableInterface.syncvars!(problem::Model, f::Union{SingleVarCostFunc, SingleVarCostGradHess}, x)
    @inbounds problem.variables[f.unfixed] = x
    return problem
end
function OptimizationVariableInterface.syncvars!(problem::Model, f::Union{MultiVarCostFunc, MultiVarDenseFDF, MultiVarSparseFDF, MultiVarSchurFDF}, x)
    if isnothing(f.unfixed_inds)
        problem.variables === x || copyto!(problem.variables, x)
    else
        sync_unfixed_vars!(problem, f.unfixed_inds, x)
    end
    return problem
end

# Number of non-zeros in the Cholesky factor predicted by CHOLMOD's symbolic analysis of an `n`-by-`n` matrix.
# Reads CHOLMOD's internal struct, so it validates what it finds and returns `nothing` if anything looks wrong.
function cholmod_symbolic_nnz(fac::SparseArrays.CHOLMOD.Factor, n::Integer)
    try
        fac.ptr == C_NULL && return nothing
        s = unsafe_load(fac.ptr)
        (s.ColCount == C_NULL || Int(s.n) != n) && return nothing
        colcount = unsafe_wrap(Array, Ptr{Int64}(s.ColCount), n)
        all(c -> 1 <= c <= n, colcount) || return nothing
        return Int(sum(colcount))
    catch
        return nothing
    end
end

# Dense or sparse storage of the reduced system, the one rule of the normal and the tall path. `storage` forces it (a dense
# one above `max_dense_bytes` is an `ArgumentError` naming `what`); `:auto` chooses dense for a reduced system of at most
# 2 000 that fits under the cap and whose Cholesky factor CHOLMOD predicts to be at least 40% full. Returns
# `(is_dense, pattern, analysed)`: the reduced system's pattern, when it was built (for the decision, or because the caller
# `needpattern`), and CHOLMOD's symbolic analysis of it, when that was run.
function reduced_storage(structure::SchurStructure, extra_blocks, storage::Symbol, max_dense_bytes::Int, what::String; needpattern::Bool)
    if storage === :dense
        check_densefits(what, structure.nr, structure.nr, max_dense_bytes)
        return true, nothing, nothing
    end
    pattern = (needpattern || storage === :auto) ? build_schur_pattern(structure; extra_blocks) : nothing
    if storage === :auto && structure.nr <= 2000 && densefits(structure.nr, structure.nr, max_dense_bytes)
        chol_fac = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Symmetric(pattern.S_sparse, :U)))
        # Fall back to the pattern's own size if CHOLMOD's prediction is unavailable
        factor_nnz = something(cholmod_symbolic_nnz(chol_fac, structure.nr), nnz(pattern.S_sparse))
        return factor_nnz >= 0.40 * 0.5 * structure.nr^2, pattern, chol_fac
    end
    return false, pattern, nothing
end

function optimvars(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction::Symbol=:auto, storage::Symbol=:auto,
        max_dense_bytes::Integer=MAX_DENSE_BYTES, min_work::Integer=MIN_WORK_PER_THREAD_NORMAL,
        min_work_cost::Integer=MIN_WORK_PER_THREAD_COST)
    check_reduction(reduction)
    check_storage(storage)
    return optimvars_dispatch(problem, convertunfixed(unfixed, problem), static(numthreads), reduction, storage, Int(max_dense_bytes), min_work, min_work_cost)
end

function optimvars_dispatch(problem::Model, unfixed::Integer, numthreads::StaticInt, reduction::Symbol, storage::Symbol, max_dense_bytes::Int, min_work::Integer, min_work_cost::Integer)
    # One variable's system is always dense (or static)
    storage === :sparse && throw(ArgumentError("storage = :sparse is not available for a single unfixed variable: its Hessian is always dense; use :auto or :dense."))
    x0 = problem.variables[unfixed]
    varlen = nvars(x0)
    check_densefits("Hessian of a single variable", dynamic(varlen), dynamic(varlen), max_dense_bytes, "raise max_dense_bytes")
    if dynamic(is_static(varlen)) && varlen <= 16
        ls = LinearSystem{UniVariateLSstatic_x{dynamic(varlen)}, UniVariateLSstatic_ls{dynamic(varlen), dynamic(varlen * varlen)}}((unfixed,), ())
    else
        ls = LinearSystem{UniVariateLSdynamic_x, UniVariateLSdynamic_ls}((unfixed, dynamic(varlen)), (dynamic(varlen),))
    end
    ls_shared = sharedlinearsystem(ls, workestimates(problem, unfixed).normal, numthreads, costsplit(problem.costs), min_work)
    f = SingleVarCostFunc(problem, unfixed, numthreads, min_work_cost)
    fdf! = SingleVarCostGradHess(ls_shared, problem, unfixed)
    # The Hessian prototype has the type of the system's own storage (an MMatrix or a Matrix), so buffers built from it
    # take fdf!'s direct-write branch
    H = fill!(copy_structure(ls.ls).A, 0.0)
    return (f, fdf!, x0, H)
end

function optimvars_dispatch(problem::Model, unfixed::AbstractVector, numthreads::StaticInt, reduction::Symbol, storage::Symbol, max_dense_bytes::Int, min_work::Integer, min_work_cost::Integer)
    @assert length(problem.variables) > 0
    nblocks = sum(unfixed)
    if nblocks == 1
        return optimvars_dispatch(problem, findfirst(unfixed), numthreads, reduction, storage, max_dense_bytes, min_work, min_work_cost)
    end

    partition = schur_partition(problem, unfixed, reduction)
    if !isnothing(partition)
        structure = SchurStructure(partition.r_dims, partition.e_dims, partition.couplings; nchunks=dynamic(numthreads))

        # Dense or sparse. The pattern includes retained-retained pairs sharing a cost, not only those sharing an
        # eliminated variable. When the symbolic analysis is run for the decision, it is kept and handed on with the
        # pattern, so the linear system does not analyse the same pattern again.
        is_dense, pattern, analysed = reduced_storage(structure, partition.extra_blocks, storage, max_dense_bytes, "reduced Hessian"; needpattern=true)

        total_e_size = sum(Int(d)^2 for d in structure.e_dims; init=0)
        total_c_size = sum(Int(structure.e_dims[e]) * Int(structure.r_dims[structure.entry_retained[c]]) for e in eachindex(structure.e_dims) for c in structure.eliminated_ptr[e]:(structure.eliminated_ptr[e+1]-1); init=0)
        E_data = zeros(Float64, total_e_size)
        C_data = zeros(Float64, total_c_size)

        H = if is_dense
            R_dense = zeros(Float64, structure.nr, structure.nr)
            SchurBlockMatrix(structure, R_dense, E_data, C_data)
        else
            R_sys = SparseBlockLinearSystem{Float64}(TransposedPattern(pattern.block_pattern), structure.r_dims, structure.r_dims)
            # The pattern (and analysed factor, if any) travel with the matrix to the linear system
            carried = isnothing(analysed) ? pattern : merge(pattern, (S_factor = analysed,))
            SchurBlockMatrix(structure, R_sys, E_data, C_data; pattern=carried)
        end

        nblocks_tot = partition.nrblocks + partition.neblocks
        blockindices = zeros(UInt, length(problem.variables))
        for (b, var_idx) in enumerate(partition.unfixed_inds)
            blockindices[var_idx] = b
        end

        boffsets = Vector{Int}(undef, nblocks_tot + 1)
        for b in 1:partition.nrblocks
            boffsets[b] = structure.boffsets_r[b]
        end
        for e in 1:partition.neblocks
            boffsets[partition.nrblocks + e] = structure.nr + structure.boffsets_e[e]
        end
        boffsets[end] = structure.nr + sum(Int, structure.e_dims; init=0) + 1

        tot_len = structure.nr + sum(Int, structure.e_dims; init=0)
        x_vec = zeros(Float64, tot_len)
        b_vec = zeros(Float64, tot_len)

        ls_x = MultiVariateLSSchur_x(x_vec, blockindices, boffsets, structure, partition.unfixed_inds)
        ls_val = MultiVariateLSSchur_ls(H, b_vec)
        lsschur = LinearSystem(ls_x, ls_val)
        ls_shared = sharedschurlinearsystem(lsschur, workestimates(problem, partition.unfixed_inds).normal, numthreads, costsplit(problem.costs), min_work)
        plan = ls_shared isa LinearSystemShared ? build_schur_thread_plan!(problem, blockindices, partition.nrblocks, partition.neblocks, length(ls_shared.ls)) : nothing

        f = MultiVarCostFunc(problem, partition.unfixed_inds, numthreads, min_work_cost)
        fdf! = MultiVarSchurFDF(ls_shared, problem, partition.unfixed_inds, plan)
        x0 = problem.variables[partition.unfixed_inds]
        return (f, fdf!, x0, H)
    end

    blocksizes, blockindices, unfixed_inds, is_natural = block_sizes_indices(problem.variables, unfixed)
    len = Int(sum(blocksizes))
    if all(unfixed) && is_natural
        unfixed_sync = nothing
        x0 = problem.variables
    else
        unfixed_sync = unfixed_inds
        x0 = problem.variables[unfixed_inds]
    end
    f = MultiVarCostFunc(problem, unfixed_sync, numthreads, min_work_cost)

    # storage forces the choice; :auto takes the structure's, but never a dense Hessian above max_dense_bytes
    storage === :dense && check_densefits("Hessian", len, len, max_dense_bytes)
    sparsity = block_sparsity(problem, unfixed_inds)
    issparse = storage === :sparse ||
        (storage === :auto && (!densefits(len, len, max_dense_bytes) || (len >= 40 && sparse_dense_decision(len, sparsity, blocksizes))))
    if issparse
        bsm = SparseBlockLinearSystem{Float64}(TransposedPattern(sparsity), blocksizes, blocksizes)
        lssparse = MultiVariateLSsparse(bsm, blockindices, true)
        ls_shared = sharedlinearsystem(lssparse, workestimates(problem, unfixed).normal, numthreads, costsplit(problem.costs), min_work)
        fdf! = MultiVarSparseFDF(ls_shared, problem, unfixed_sync)
        return (f, fdf!, x0, lssparse.x.hessian)
    else
        lsdense = MultiVariateLSdense(blocksizes, blockindices)
        ls_shared = sharedlinearsystem(lsdense, workestimates(problem, unfixed).normal, numthreads, costsplit(problem.costs), min_work)
        fdf! = MultiVarDenseFDF(ls_shared, problem, unfixed_sync)
        return (f, fdf!, x0, nothing)
    end
end

function NLPLSModel end

NLPLSModel(problem::Model, args...; kwargs...) =
    error("NLPLSModel requires the NLPModels package. Run `using NLPModels` first.")

function NLLSNLSModel end

NLLSNLSModel(problem::Model, args...; kwargs...) =
    error("NLLSNLSModel requires the NLPModels package. Run `using NLPModels` first.")
