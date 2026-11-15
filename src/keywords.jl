# The keywords shared by every way of building or optimizing a model: `optimvars`, `optimvars_nls`, `flat_optimvars_nls`,
# `optimize!(model, options; ...)`, `NLPLSModel` and `NLLSNLSModel` take the same set, with the same names, meanings and
# defaults. Their checks and their description live here, once.

const MAX_DENSE_BYTES = 256 * 1024^2
const REDUCTION_VALUES = (:auto, :none, :eliminate)
const STORAGE_VALUES = (:auto, :dense, :sparse)

function check_reduction(reduction::Symbol)
    reduction in REDUCTION_VALUES || throw(ArgumentError("Unknown reduction $(repr(reduction)); expected :auto, :none or :eliminate."))
    return reduction
end
function check_storage(storage::Symbol)
    storage in STORAGE_VALUES || throw(ArgumentError("Unknown storage $(repr(storage)); expected :auto, :dense or :sparse."))
    return storage
end

# A dense `rows`-by-`cols` matrix of Float64 fits under the cap
densefits(rows::Integer, cols::Integer, max_dense_bytes::Integer) = 8 * Int(rows) * Int(cols) <= max_dense_bytes
function check_densefits(what, rows::Integer, cols::Integer, max_dense_bytes::Integer, remedy="raise max_dense_bytes or use storage = :sparse")
    densefits(rows, cols, max_dense_bytes) && return nothing
    throw(ArgumentError("A dense $(rows)×$(cols) $what needs $(8 * Int(rows) * Int(cols)) bytes, which exceeds max_dense_bytes = $max_dense_bytes; $remedy."))
end

# The description of the set, interpolated into the docstring of each entry point
const KEYWORDS_DOC = """
# Keywords

The same for [`optimvars`](@ref), [`optimvars_nls`](@ref), [`flat_optimvars_nls`](@ref), `optimize!(model, options)`,
[`NLPLSModel`](@ref) and [`NLLSNLSModel`](@ref).

- `unfixed = nothing`: which variables to optimize: a variable index, a variable type, a Boolean mask or a list of
  indices. `nothing` optimizes all of them; the others are left untouched.
- `numthreads = StaticInt(Threads.nthreads())`: the most threads an evaluation uses (`StaticInt` or `Integer`).
- `reduction = :auto`: whether eliminable variables are eliminated (the Schur complement). `:auto` eliminates when the
  model has `iseliminable` variables and the structure fits, and otherwise does not; `:eliminate` eliminates or throws an
  `ArgumentError`; `:none` never eliminates. On the tall path (the residual and Jacobian) `:auto` means `:none`, and
  `:eliminate` is an `ArgumentError`: elimination is not implemented there.
- `storage = :auto`: storage of the matrix the solver receives: the Hessian, the reduced Hessian under elimination, or
  the Jacobian. `:auto` chooses from the structure (the sparsity of the Hessian, the predicted fill of the reduced
  system's Cholesky factor, or the fill of the Jacobian); `:dense` and `:sparse` force the choice. A single unfixed
  variable's Hessian is always dense, so `:sparse` is an `ArgumentError` there.
- `max_dense_bytes = $MAX_DENSE_BYTES` (256 MiB): under `storage = :auto` a dense matrix larger than this is not chosen (it is
  stored sparse instead); under `:dense` a larger one is an `ArgumentError` saying how large it would be, raised before it
  is allocated. A single variable's Hessian has no sparse alternative, so above the cap it is an `ArgumentError` either way.
- `min_work`: per-thread floor of the fill, in the work estimate of the formulation: on the normal path (the normal fill,
  forming `JᵀJ` and `Jᵀr`; default $MIN_WORK_PER_THREAD_NORMAL) one number written, per cost `m * n * (n + 1)`; on the tall path (the
  tall fill, writing `r` and `J`; default $MIN_WORK_PER_THREAD_TALL) one entry of `r` or `J` written, per cost `m * (n + 1)`.
- `min_work_cost = $MIN_WORK_PER_THREAD_COST`: per-thread floor of a cost evaluation, one residual entry, per cost `m`.

Each kind of work is split over `clamp(work ÷ floor, 1, numthreads)` threads: one thread when its estimate is below
twice the floor, and every thread at a floor of 0. The costs of each type are split into one chunk per thread, so the
count is also never more than the largest number of costs of one type: a thread beyond that would have nothing to do. The normal fill uses at most $MAX_THREADS_NORMAL threads: each thread fills its own
copy of the linear system, which costs work in proportion to the size of the system, so on large systems more threads
were measured to be slower (the number of copies is also capped by memory). The defaults were measured on cheap residuals,
on one machine; a costly residual could thread earlier than they let it.

Here `m` is a cost's residual length (1 for a general [`AbstractCost`](@ref)) and `n` the total dimension of its
unfixed variables; each estimate is summed over all costs and computed once, at setup. The work of the residual
functions themselves is not estimated.
"""
