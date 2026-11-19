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
# The wrappers that hand a plain Jacobian to external solvers (`flat_optimvars_nls`, `nls_callbacks`, `NLLSNLSModel`) cannot
# eliminate: `:auto` does not eliminate there, and `:eliminate` is refused up front. Returns the reduction to build with.
function check_plain_reduction(reduction::Symbol, name)
    check_reduction(reduction)
    reduction === :eliminate && throw(ArgumentError("$name hands a plain Jacobian to external solvers, which cannot eliminate variables: reduction = :eliminate is not supported; use :auto or :none (neither eliminates here)."))
    return :none
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

# What `formulation = :jacobian` supports, interpolated into the docstrings of the front door (`optimize!(model, options)`) and
# of `optimvars_nls`. The README and docs/src/index.md hold the same table, by hand.
const JACOBIAN_FORMULATION_DOC = """
| `reduction` | `storage` | Jacobian prototype | `linsolver` |
|:---|:---|:---|:---|
| no elimination | dense | `Matrix{Float64}` | `:auto`, `:qr`; `:cholesky`, `:ldl` (the normal equations, formed from `J`); `:lsmr` (with Krylov.jl) |
| no elimination | sparse | `SparseMatrixCSC{Float64, Int}` | `:auto`, `:qr`; `:lsmr` (with Krylov.jl) |
| elimination | dense | `SchurBlockJacobian{Float64, Matrix{Float64}}` | `:auto`, `:qr`, `:cholesky`, `:ldl`; `:lsmr` (with Krylov.jl) |
| elimination | sparse | `SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}` | `:auto`, `:qr`, `:cholesky`, `:ldl`; `:lsmr` (with Krylov.jl) |

`reduction` and `storage` are as for the Hessian path, `:auto` included: elimination when the model has eliminable
variables and the structure fits (on the Jacobian path also: each eliminated variable is in a cost), the reduced system
stored by the Hessian path's rule, and without elimination the Jacobian's storage from its fill, never dense above
`max_dense_bytes`. The `linsolver` values are those SecondOrderOptimization accepts for each prototype: its rules,
the linear system of each row and the iterators it takes are in the "Supported combinations" table of its
`JacobianObjective` docstring. Any other combination is an `ArgumentError` before anything is evaluated.
"""

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
  `ArgumentError`; `:none` never eliminates. The rule is the same on the Jacobian path (the residual and Jacobian, where the
  eliminated variables are eliminated by QR), which also needs every eliminated variable to be in a cost. The wrappers
  that hand a plain Jacobian to external solvers (`flat_optimvars_nls`, `NLLSNLSModel`) never eliminate: there `:auto`
  does not eliminate and `:eliminate` is an `ArgumentError`. A single unfixed variable is never eliminated.
- `storage = :auto`: storage of the matrix the solver receives: the Hessian, the reduced system under elimination (on
  either path), or the Jacobian. `:auto` chooses from the structure (the sparsity of the Hessian, the predicted fill of the reduced
  system's Cholesky factor, or the fill of the Jacobian); `:dense` and `:sparse` force the choice. On the Jacobian path the
  Jacobian is a `Matrix{Float64}` or a `SparseMatrixCSC{Float64, Int}`, and `optimize!(...; formulation = :jacobian)` hands
  either to the optimizer. On the Hessian path a single unfixed variable's Hessian is always dense, so `:sparse` is an
  `ArgumentError` there.
- `max_dense_bytes = $MAX_DENSE_BYTES` (256 MiB): under `storage = :auto` a dense matrix larger than this is not chosen (it is
  stored sparse instead); under `:dense` a larger one is an `ArgumentError` saying how large it would be, raised before it
  is allocated. A single variable's Hessian has no sparse alternative, so above the cap it is an `ArgumentError` either way.
- `min_work`: per-thread floor of the fill, in the work estimate of the formulation: on the Hessian path (the Hessian fill,
  forming `JᵀJ` and `Jᵀr`; default $MIN_WORK_PER_THREAD_HESSIAN) one number written, per cost `m * n * (n + 1)`; on the Jacobian path (the
  Jacobian fill, writing `r` and `J`; default $MIN_WORK_PER_THREAD_JACOBIAN) one entry of `r` or `J` written, per cost `m * (n + 1)`.
- `min_work_cost = $MIN_WORK_PER_THREAD_COST`: per-thread floor of a cost evaluation, one residual entry, per cost `m`.

Each kind of work is split over `clamp(work ÷ floor, 1, numthreads)` threads: one thread when its estimate is below
twice the floor, and every thread at a floor of 0. The costs of each type are split into one chunk per thread, so the
count is also never more than the largest number of costs of one type: a thread beyond that would have nothing to do. The Hessian fill uses at most $MAX_THREADS_HESSIAN threads: each thread fills its own
copy of the linear system, which costs work in proportion to the size of the system, so on large systems more threads
were measured to be slower (the number of copies is also capped by memory). The defaults were measured on cheap residuals,
on one machine; a costly residual could thread earlier than they let it.

Here `m` is a cost's residual length (1 for a general [`AbstractCost`](@ref)) and `n` the total dimension of its
unfixed variables; each estimate is summed over all costs and computed once, at setup. The work of the residual
functions themselves is not estimated.
"""
