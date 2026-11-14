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
  the Jacobian. `:auto` chooses from the structure; `:dense` and `:sparse` force the choice. On the normal path with
  nothing eliminated it is not used: the structure decides.
- `max_dense_bytes = $MAX_DENSE_BYTES` (256 MiB): on the tall path, under `storage = :auto`, a dense Jacobian larger than this is not
  chosen.
- `min_work`: threading floor of the linearization, in the work estimate of the formulation: on the normal path
  (default $MIN_WORK_FOR_THREADED_LINEARIZATION) the linearization is threaded only if its estimate is at least this, one number
  written in forming the normal equations, per cost `m * n * (n + 1)`; on the tall path (default $MIN_WORK_PER_THREAD) the residual
  and Jacobian fill is threaded only if its estimate is at least twice this, on at most `work ÷ min_work` threads, one
  entry of `r` or `J` written, per cost `m * (n + 1)`.
- `min_work_cost = $MIN_WORK_FOR_THREADED_COST`: a cost evaluation is threaded only if its estimate is at least this, one residual
  entry, per cost `m`.

Here `m` is a cost's residual length (1 for a general [`AbstractCost`](@ref)) and `n` the total dimension of its
unfixed variables; each estimate is summed over all costs and computed once, at setup. The work of the residual
functions themselves is not estimated.
"""
