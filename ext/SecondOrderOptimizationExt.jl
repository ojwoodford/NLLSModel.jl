module SecondOrderOptimizationExt

using NLLSModel
using SecondOrderOptimization
using OptimizationVariableInterface
using Static
import NLLSModel: MIN_WORK_PER_THREAD_NORMAL, MIN_WORK_PER_THREAD_COST, MIN_WORK_PER_THREAD_TALL, MAX_DENSE_BYTES, KEYWORDS_DOC

# Hands `resjac!` to the optimizer with `prezeroed = true`: the structural zeros of `J` are already in place (the optimizer
# allocates `J` as zeros and, from SecondOrderOptimization's Z1, neither re-zeroes it nor writes anything but this evaluator's
# entries), and the evaluator writes every entry that can be non-zero on every call, so it need not fill `J` first.
# A struct rather than a closure, so the objective's type is concrete.
struct TallResJac{RJ}
    resjac!::RJ
end
(w::TallResJac)(r, J, x) = w.resjac!(r, J, x; prezeroed=true)

# The tall formulation: the residual vector and a dense Jacobian, solved through QR of the Jacobian.
function optimize_tall!(model::Model, options; unfixed, numthreads, callback, workspace, reduction, storage, max_dense_bytes, min_work, min_work_cost)
    # The optimizer solves through QR of a dense J
    storage === :sparse && throw(ArgumentError("formulation = :tall needs a dense Jacobian (the optimizer solves through QR of a dense J); storage = :sparse is not supported, use :auto or :dense."))

    # Refuse before anything is allocated: the tall formulation holds the whole Jacobian
    if all(cv -> eltype(cv) <: AbstractResidual, values(model.costs)) # optimvars_nls reports other costs
        NLLSModel.check_densefits("Jacobian", NLLSModel.nresiduals(model), NLLSModel.unfixeddims(model, NLLSModel.convertunfixed(unfixed, model)),
            max_dense_bytes, "formulation = :tall needs a dense Jacobian; raise max_dense_bytes or use formulation = :normal")
    end
    f, _, resjac!, x0, J = try
        optimvars_nls(model; unfixed, numthreads, reduction, storage=:dense, max_dense_bytes, min_work, min_work_cost)
    catch err
        err isa ArgumentError ? throw(ArgumentError("formulation = :tall: " * err.msg)) : rethrow()
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
    optimize!(model::Model, options = OptimizationOptions(); formulation=:normal, callback=nullcallback, workspace=nothing, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work, min_work_cost=$MIN_WORK_PER_THREAD_COST) -> OptimizationResult

Solve `model` and write the solution back into `model.variables`. This is the normal way to solve a `Model`; it is defined when
`SecondOrderOptimization` is loaded (`using SecondOrderOptimization`).

`formulation` chooses how the model is linearized:

- `:normal` (default): the normal equations. Calls [`optimvars`](@ref), wraps the result in a `HessianObjective`, runs
  `SecondOrderOptimization.optimize!` (or `optimize` when the variable to optimize is immutable, a single variable of static
  size) and writes the solution back with [`syncvars!`](@ref).
- `:tall`: the residual vector and a dense Jacobian (a `JacobianObjective`), solved through QR of the Jacobian, with any
  iterator except gradient descent. Calls [`optimvars_nls`](@ref) with `storage = :dense`. Each cost's kernel decides how its rows enter; there is no robust
  keyword. This first version needs `linsolver = :auto` or `:qr` (the optimizer throws `ArgumentError` for gradient
  descent and for other solvers), a dense Jacobian (`storage = :sparse` is an `ArgumentError`, and so is a Jacobian larger
  than `max_dense_bytes`, before allocating), only non-adaptive `AbstractResidual` costs and no unfixed adaptive kernels
  (errors from `optimvars_nls` are rethrown prefixed `formulation = :tall:`), and no elimination.

`formulation` is a keyword here only: it is not a field of `OptimizationOptions` and is not inferred from `options.linsolver`. Any
other value is an `ArgumentError`. `callback` and `workspace` are passed to the optimizer. The default of `min_work` is the
formulation's ($MIN_WORK_PER_THREAD_NORMAL for `:normal`, $MIN_WORK_PER_THREAD_TALL for `:tall`).

$KEYWORDS_DOC"""
function SecondOrderOptimization.optimize!(model::Model, options::OptimizationOptions=OptimizationOptions();
        formulation::Symbol=:normal, callback=nullcallback, workspace=nothing,
        unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction::Symbol=:auto, storage::Symbol=:auto,
        max_dense_bytes::Integer=MAX_DENSE_BYTES,
        min_work::Integer=(formulation === :tall ? MIN_WORK_PER_THREAD_TALL : MIN_WORK_PER_THREAD_NORMAL),
        min_work_cost::Integer=MIN_WORK_PER_THREAD_COST)
    formulation in (:normal, :tall) || throw(ArgumentError("Unknown formulation $(repr(formulation)); expected :normal or :tall."))
    NLLSModel.check_reduction(reduction)
    NLLSModel.check_storage(storage)
    formulation === :tall && return optimize_tall!(model, options; unfixed, numthreads, callback, workspace, reduction, storage, max_dense_bytes, min_work, min_work_cost)
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

end
