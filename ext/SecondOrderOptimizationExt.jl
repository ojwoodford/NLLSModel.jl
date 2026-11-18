module SecondOrderOptimizationExt

using NLLSModel
using SecondOrderOptimization
using OptimizationVariableInterface
using Static
import NLLSModel: MIN_WORK_PER_THREAD_NORMAL, MIN_WORK_PER_THREAD_COST, MIN_WORK_PER_THREAD_TALL, MAX_DENSE_BYTES, KEYWORDS_DOC, TALL_FORMULATION_DOC,
    OptimizationRecommendation

# Hands `resjac!` to the optimizer with `prezeroed = true`: the structural zeros of `J` are already in place (the optimizer
# allocates `J` as zeros and, from SecondOrderOptimization's Z1, neither re-zeroes it nor writes anything but this evaluator's
# entries), and the evaluator writes every entry that can be non-zero on every call, so it need not fill `J` first.
# A struct rather than a closure, so the objective's type is concrete.
struct TallResJac{RJ}
    resjac!::RJ
end
(w::TallResJac)(r, J, x) = w.resjac!(r, J, x; prezeroed=true)

# The tall formulation: the residual vector and the Jacobian, solved by the optimizer from the Jacobian prototype that
# `optimvars_nls` returns (its type decides the linear system; the optimizer validates it against `linsolver`). A dense
# Jacobian forced above `max_dense_bytes` is refused by `optimvars_nls` before it is allocated.
function optimize_tall!(model::Model, options; unfixed, numthreads, callback, workspace, reduction, storage, max_dense_bytes, min_work, min_work_cost)
    f, _, resjac!, x0, J = try
        optimvars_nls(model; unfixed, numthreads, reduction, storage, max_dense_bytes, min_work, min_work_cost)
    catch err
        err isa ArgumentError ? throw(ArgumentError("formulation = :jacobian: " * err.msg)) : rethrow()
    end
    obj = JacobianObjective(f, TallResJac(resjac!), J)
    if ismutablevar(x0)
        result = SecondOrderOptimization.optimize!(x0, obj, options; callback, workspace)
        syncvars!(model, resjac!, x0)
    else
        x, result = SecondOrderOptimization.optimize(x0, obj, options; callback, workspace)
        syncvars!(model, resjac!, x)
    end
    return result
end

# Solves the model with the normal equations (`HessianObjective`). A single variable of static size is immutable, so it goes
# through `optimize` (which returns the solution); everything else is updated in place by `optimize!`. Either way the
# solution is written back with `syncvars!`, which knows the block order and the Schur partition.
"""
    optimize!(model::Model, options = OptimizationOptions(); formulation=:hessian, callback=nullcallback, workspace=nothing, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work, min_work_cost=$MIN_WORK_PER_THREAD_COST) -> OptimizationResult

Solve `model` and write the solution back into `model.variables`. This is the normal way to solve a `Model`; it is defined when
`SecondOrderOptimization` is loaded (`using SecondOrderOptimization`).

The second method, `optimize!(model, rec)`, takes every choice from an `OptimizationRecommendation` made by
[`recommendoptimizer`](@ref); its docstring is below.

`formulation` chooses how the model is linearized:

- `:hessian` (default): the normal equations. Calls [`optimvars`](@ref), wraps the result in a `HessianObjective`, runs
  `SecondOrderOptimization.optimize!` (or `optimize` when the variable to optimize is immutable, a single variable of static
  size) and writes the solution back with [`syncvars!`](@ref).
- `:jacobian`: the residual vector and the Jacobian (a `JacobianObjective`); the whole `JᵀJ` is never formed.
  Calls [`optimvars_nls`](@ref) with the same keywords and hands its Jacobian prototype to the optimizer, which chooses the
  linear system from it; the table below. Each cost's kernel decides how its rows enter; there is no robust keyword. Every
  cost must be a non-adaptive `AbstractResidual`, with no unfixed adaptive kernel; errors from `optimvars_nls` are
  rethrown prefixed `formulation = :jacobian:`, and a dense Jacobian forced above `max_dense_bytes` is one of them, raised
  before it is allocated.

$TALL_FORMULATION_DOC

`formulation` is a keyword here only: it is not a field of `OptimizationOptions` and is not inferred from `options.linsolver`. Any
other value is an `ArgumentError`. `callback` and `workspace` are passed to the optimizer. The default of `min_work` is the
formulation's ($MIN_WORK_PER_THREAD_NORMAL for `:hessian`, $MIN_WORK_PER_THREAD_TALL for `:jacobian`).

$KEYWORDS_DOC"""
function SecondOrderOptimization.optimize!(model::Model, options::OptimizationOptions=OptimizationOptions();
        formulation::Symbol=:hessian, callback=nullcallback, workspace=nothing,
        unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction::Symbol=:auto, storage::Symbol=:auto,
        max_dense_bytes::Integer=MAX_DENSE_BYTES,
        min_work::Integer=(formulation === :jacobian ? MIN_WORK_PER_THREAD_TALL : MIN_WORK_PER_THREAD_NORMAL),
        min_work_cost::Integer=MIN_WORK_PER_THREAD_COST)
    formulation in (:hessian, :jacobian) || throw(ArgumentError("Unknown formulation $(repr(formulation)); expected :hessian or :jacobian."))
    NLLSModel.check_reduction(reduction)
    NLLSModel.check_storage(storage)
    formulation === :jacobian && return optimize_tall!(model, options; unfixed, numthreads, callback, workspace, reduction, storage, max_dense_bytes, min_work, min_work_cost)
    f, fdf!, x0, H = optimvars(model; unfixed, numthreads, reduction, storage, max_dense_bytes, min_work, min_work_cost)
    obj = HessianObjective(f, fdf!; H)
    if ismutablevar(x0)
        result = SecondOrderOptimization.optimize!(x0, obj, options; callback, workspace)
        syncvars!(model, fdf!, x0)
    else
        x, result = SecondOrderOptimization.optimize(x0, obj, options; callback, workspace)
        syncvars!(model, fdf!, x)
    end
    return result
end

"""
    optimize(model::Model, options = OptimizationOptions(); kwargs...) -> (model_copy, result)

Like `optimize!(model, options; kwargs...)` (including `formulation`), but runs on a `deepcopy` of `model` and leaves `model` unchanged.
"""
function SecondOrderOptimization.optimize(model::Model, options::OptimizationOptions=OptimizationOptions(); kwargs...)
    copied = deepcopy(model)
    result = SecondOrderOptimization.optimize!(copied, options; kwargs...)
    return copied, result
end

# The keywords a recommendation sets: the options' iterator and linear solver, the formulation, and the keywords of the
# model's linearization that it was made with
const RECOMMENDATION_KEYWORDS = (:iterator, :linsolver, :formulation, :reduction, :storage, :unfixed, :numthreads, :max_dense_bytes)

"""
    optimize!(model::Model, rec::OptimizationRecommendation; callback=nullcallback, workspace=nothing, min_work, min_work_cost=$MIN_WORK_PER_THREAD_COST, options...) -> OptimizationResult

Solve `model` as the recommendation `rec` (from [`recommendoptimizer`](@ref)) says, and write the solution back into
`model.variables`. It builds `OptimizationOptions(; iterator = rec.iterator, linsolver = rec.linsolver, options...)` and
calls `optimize!(model, options; formulation, reduction, storage, unfixed, numthreads, max_dense_bytes)` with the
recommendation's values: exactly the call that displaying `rec` prints.

The remaining keywords are passed to the `OptimizationOptions` constructor (`maxiters`, `maxtime`, the tolerances, ...),
so an unknown one is an error there. A keyword the recommendation sets (`iterator`, `linsolver`, `formulation`,
`reduction`, `storage`, `unfixed`, `numthreads`, `max_dense_bytes`) is an `ArgumentError`, raised before anything is
built: to change one, use `optimize!(model, options; ...)`, starting from the call that `show` prints. `callback`,
`workspace`, `min_work` (default the formulation's) and `min_work_cost` are as for `optimize!(model, options)`.

The options are built from the recommendation's values at run time, so the call dispatches once on their type. Setting
the options by hand, from the call that `show` prints, avoids it.
"""
function SecondOrderOptimization.optimize!(model::Model, rec::OptimizationRecommendation;
        callback=nullcallback, workspace=nothing,
        min_work::Integer=(rec.formulation === :jacobian ? MIN_WORK_PER_THREAD_TALL : MIN_WORK_PER_THREAD_NORMAL),
        min_work_cost::Integer=MIN_WORK_PER_THREAD_COST, options...)
    # The one slurped keyword list in an entry point (rules.md 19): every keyword in it goes to the `OptimizationOptions`
    # constructor, whose keywords are spelled out, so a misspelt or stale keyword is still an error
    for name in keys(options)
        name in RECOMMENDATION_KEYWORDS && throw(ArgumentError("$name is set by the recommendation; to set it by hand, use optimize!(model, options; ...), starting from the call that displaying the recommendation prints."))
    end
    opts = OptimizationOptions(; iterator=rec.iterator, linsolver=rec.linsolver, options...)
    return SecondOrderOptimization.optimize!(model, opts; formulation=rec.formulation, reduction=rec.reduction,
        storage=rec.storage, unfixed=rec.unfixed, numthreads=rec.numthreads, max_dense_bytes=rec.max_dense_bytes,
        callback, workspace, min_work, min_work_cost)
end

"""
    optimize(model::Model, rec::OptimizationRecommendation; kwargs...) -> (model_copy, result)

Like `optimize!(model, rec; kwargs...)`, but runs on a `deepcopy` of `model` and leaves `model` unchanged.
"""
function SecondOrderOptimization.optimize(model::Model, rec::OptimizationRecommendation; kwargs...)
    copied = deepcopy(model)
    result = SecondOrderOptimization.optimize!(copied, rec; kwargs...)
    return copied, result
end

end
