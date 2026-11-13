module SecondOrderOptimizationExt

using NLLSModel
using SecondOrderOptimization
using OptimizationVariableInterface
using Static
import NLLSModel: MIN_WORK_FOR_THREADED_LINEARIZATION, MIN_WORK_FOR_THREADED_COST

# Hands `resjac!` to the optimizer. The optimizer zeroes `r` and `J` before every call, so the evaluator need not.
# A struct rather than a closure, so the objective's type is concrete.
struct PrezeroedResJac{RJ}
    resjac!::RJ
end
(w::PrezeroedResJac)(r, J, x) = w.resjac!(r, J, x; prezeroed=true)

# The tall formulation: the residual vector and a dense Jacobian, solved by Levenberg-Marquardt through QR of the Jacobian.
function optimize_tall!(model::Model, options; unfixed, numthreads, callback, workspace, reduction, min_work_per_thread, max_dense_bytes)
    reduction in (:auto, :none, :eliminate) || throw(ArgumentError("Unknown reduction $(repr(reduction)); expected :auto, :none or :eliminate."))
    reduction === :eliminate && throw(ArgumentError("formulation = :tall does not support reduction = :eliminate (elimination of variables on the Jacobian path is not implemented); use :auto or :none."))

    # Refuse before anything is allocated: the tall formulation holds the whole Jacobian
    mask = NLLSModel.convertunfixed(unfixed, model)
    n = mask isa Integer ? Int(dynamic(nvars(model.variables[mask]))) : sum(i -> Int(dynamic(nvars(model.variables[i]))), findall(mask); init=0)
    m = all(cv -> eltype(cv) <: AbstractResidual, values(model.costs)) ? nresiduals(model) : 0 # optimvars_nls reports other costs
    if all(cv -> eltype(cv) <: AbstractResidual, values(model.costs)) && 8 * m * n > max_dense_bytes
        throw(ArgumentError("formulation = :tall needs a dense $m×$n Jacobian ($(8 * m * n) bytes), which exceeds max_dense_bytes = $max_dense_bytes. The tall formulation currently needs a dense Jacobian; raise max_dense_bytes or use formulation = :normal."))
    end

    f, _, resjac!, x0, J = try
        optimvars_nls(model; unfixed, numthreads, min_work_per_thread=Int(min_work_per_thread), matrix_type=:dense, max_dense_bytes=Int(max_dense_bytes))
    catch err
        err isa ArgumentError ? throw(ArgumentError("formulation = :tall: " * err.msg)) : rethrow()
    end
    # With every variable unfixed and in natural order, x0 *is* model.variables. The evaluators copy each point they evaluate into
    # the model, which would overwrite the optimizer's current point with a trial, so the optimizer gets its own copy.
    x0 === model.variables && (x0 = copyvars(x0))
    obj = JacobianObjective(f, PrezeroedResJac(resjac!), J)
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
    optimize!(model::Model, options = OptimizationOptions(); formulation=:normal, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), callback=nullcallback, workspace=nothing, reduction=:auto, reduced=:auto, min_work, min_work_cost, min_work_per_thread=5000, max_dense_bytes=256*1024^2) -> OptimizationResult

Solve `model` and write the solution back into `model.variables`. This is the normal way to solve a `Model`; it is defined when
`SecondOrderOptimization` is loaded (`using SecondOrderOptimization`).

`formulation` chooses how the model is linearized:

- `:normal` (default): the normal equations. Calls [`optimvars`](@ref), wraps the result in a `HessianObjective`, runs
  `SecondOrderOptimization.optimize!` (or `optimize` when the variable to optimize is immutable, a single variable of static
  size) and writes the solution back with [`syncvars!`](@ref).
- `:tall`: the residual vector and a dense Jacobian (a `JacobianObjective`), solved by Levenberg–Marquardt through QR of the
  Jacobian. Calls [`optimvars_nls`](@ref) with a dense Jacobian. Each cost's kernel decides how its rows enter; there is no robust
  keyword. This first version needs an iterator the optimizer's Jacobian path supports (Levenberg–Marquardt) with `linsolver = :auto` or `:qr` (the optimizer throws
  otherwise), a dense Jacobian (an `ArgumentError`, before allocating, if it would exceed `max_dense_bytes`), only non-adaptive
  `AbstractResidual` costs and no unfixed adaptive kernels (errors from `optimvars_nls` are rethrown prefixed `formulation = :tall:`),
  and no elimination.

`formulation` is a keyword here only: it is not a field of `OptimizationOptions` and is not inferred from `options.linsolver`. Any
other value is an `ArgumentError`.

| Keyword | `:normal` | `:tall` |
|:---|:---|:---|
| `unfixed`, `numthreads`, `callback`, `workspace` | used | used |
| `reduction` | `:auto`, `:none`, `:eliminate` | `:auto` or `:none`; `:eliminate` is an `ArgumentError` |
| `reduced`, `min_work`, `min_work_cost` | used | ignored |
| `min_work_per_thread`, `max_dense_bytes` | ignored | used |

- `unfixed`: which variables to optimize: an index, a type, a Boolean mask or a list of indices (default: all). The others are left untouched.
- `numthreads`: threading of the evaluation.
- `callback`, `workspace`: passed to the optimizer.
- `reduction`, `reduced`, `min_work`, `min_work_cost`: layout and threading of the normal-equations system, as for [`optimvars`](@ref).
- `min_work_per_thread`, `max_dense_bytes`: threading and the size limit of the dense Jacobian, as for [`optimvars_nls`](@ref).
"""
function SecondOrderOptimization.optimize!(model::Model, options::OptimizationOptions=OptimizationOptions();
        formulation::Symbol=:normal,
        unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), callback=nullcallback, workspace=nothing,
        reduction::Symbol=:auto, reduced::Symbol=:auto,
        min_work::Integer=MIN_WORK_FOR_THREADED_LINEARIZATION, min_work_cost::Integer=MIN_WORK_FOR_THREADED_COST,
        min_work_per_thread::Integer=5000, max_dense_bytes::Integer=256 * 1024^2)
    formulation in (:normal, :tall) || throw(ArgumentError("Unknown formulation $(repr(formulation)); expected :normal or :tall."))
    formulation === :tall && return optimize_tall!(model, options; unfixed, numthreads, callback, workspace, reduction, min_work_per_thread, max_dense_bytes)
    f, fdf!, x0, H = optimvars(model, unfixed, numthreads; reduction, reduced, min_work, min_work_cost)
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
