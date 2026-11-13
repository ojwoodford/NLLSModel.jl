module SecondOrderOptimizationExt

using NLLSModel
using SecondOrderOptimization
using OptimizationVariableInterface
using Static
import NLLSModel: MIN_WORK_FOR_THREADED_LINEARIZATION, MIN_WORK_FOR_THREADED_COST

# Solves the model with the normal equations (`HessianObjective`). A single variable of static size is immutable, so it goes
# through `optimize` (which returns the solution); everything else is updated in place by `optimize!`. Either way the
# solution is written back with `syncvars!`, which knows the block order and the Schur partition.
"""
    optimize!(model::Model, options = OptimizationOptions(); unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), callback=nullcallback, workspace=nothing, reduction=:auto, reduced=:auto, min_work, min_work_cost) -> OptimizationResult

Solve `model` by the normal equations and write the solution back into `model.variables`. This is the normal way to solve a
`Model`; it is defined when `SecondOrderOptimization` is loaded (`using SecondOrderOptimization`).

It calls [`optimvars`](@ref), wraps the result in a `HessianObjective`, runs `SecondOrderOptimization.optimize!` (or
`optimize` when the variable to optimize is immutable, a single variable of static size) and writes the solution back
with [`syncvars!`](@ref).

- `unfixed`: which variables to optimize: an index, a type, a Boolean mask or a list of indices (default: all). The others are left untouched.
- `numthreads`, `min_work`, `min_work_cost`: threading of the evaluation, as for [`optimvars`](@ref).
- `callback`, `workspace`: passed to the optimizer.
- `reduction`, `reduced`: layout of the linear system, as for [`optimvars`](@ref). Neither is a field of `OptimizationOptions`.
"""
function SecondOrderOptimization.optimize!(model::Model, options::OptimizationOptions=OptimizationOptions();
        unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), callback=nullcallback, workspace=nothing,
        reduction::Symbol=:auto, reduced::Symbol=:auto,
        min_work::Integer=MIN_WORK_FOR_THREADED_LINEARIZATION, min_work_cost::Integer=MIN_WORK_FOR_THREADED_COST)
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

Like `optimize!(model, options; kwargs...)`, but runs on a `deepcopy` of `model` and leaves `model` unchanged.
"""
function SecondOrderOptimization.optimize(model::Model, options::OptimizationOptions=OptimizationOptions(); kwargs...)
    copied = deepcopy(model)
    result = SecondOrderOptimization.optimize!(copied, options; kwargs...)
    return copied, result
end

end
