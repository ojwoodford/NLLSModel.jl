module SecondOrderOptimizationExt

using NLLSModel
using SecondOrderOptimization
using OptimizationVariableInterface
using Static
import NLLSModel: MIN_WORK_FOR_THREADED_LINEARIZATION, MIN_WORK_FOR_THREADED_COST

# Solves the model with the normal equations (`HessianObjective`). A single variable of static size is immutable, so it goes
# through `optimize` (which returns the solution); everything else is updated in place by `optimize!`. Either way the
# solution is written back with `syncvars!`, which knows the block order and the Schur partition.
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

function SecondOrderOptimization.optimize(model::Model, options::OptimizationOptions=OptimizationOptions(); kwargs...)
    copied = deepcopy(model)
    result = SecondOrderOptimization.optimize!(copied, options; kwargs...)
    return copied, result
end

end
