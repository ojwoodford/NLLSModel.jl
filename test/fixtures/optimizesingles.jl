using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization

fixture("optimizeba", "frontdoor")

const SinglesExt = Base.get_extension(NLLSModel, :SecondOrderOptimizationExt)

const SingleOptions = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=20)

# A perturbed bundle adjustment: 10 cameras, 50 landmarks
function baproblem()
    Random.seed!(7)
    return perturb_ba_problem(create_ba_problem(10, 50, 0.3), 0.003, 0.001)
end

# The reference: each variable in `order` solved through the front door on a subproblem of its costs, the results summed
function singlesloop!(model::Model{VT, CT}, options, order; kwargs...) where {VT, CT}
    costsof = sparse(NLLSModel.getvarcostmap(model)')
    sub = Model(model.variables, CT)
    result = nothing
    for i in order
        subproblem!(sub, model, rowvals(costsof)[nzrange(costsof, i)])
        r = optimize!(sub, options; unfixed=i, numthreads=static(1), kwargs...)
        result = isnothing(result) ? r : result + r
    end
    return result
end

# The bytes a run of `indices` (one type and size) allocates, measured on a second pass after a first one has built its
# setup. The setup is built here and the run measured in a barrier that takes it: on Julia 1.10 the setup's type is
# inferred only as a union (static or dense system, chosen at run time), so a run in the same function is a dynamic call.
function singlesrun_allocs(model::Model{VT, CT}, options, indices) where {VT, CT}
    costsof = NLLSModel.costsbyvariable(model)
    sub = Model(model.variables, CT)
    subproblem!(sub, model, NLLSModel.costsof(costsof, first(indices)))
    f, fdf!, x0, H = optimvars(sub; unfixed=first(indices), numthreads=static(1))
    ws = OptimizationWorkspace(x0, HessianObjective(f, fdf!; H), options)
    result = SinglesExt.emptyresult(options)
    run = view(indices, 2:length(indices))
    return runallocs(sub, model, options, run, costsof, f, fdf!, x0, H, ws, result)
end
function runallocs(sub, model, options, run, costsof, f, fdf!, x0, H, ws, result)
    SinglesExt.optimizerun!(sub, model, options, run, costsof, f, fdf!, x0, H, ws, result)
    return @allocated SinglesExt.optimizerun!(sub, model, options, run, costsof, f, fdf!, x0, H, ws, result)
end
