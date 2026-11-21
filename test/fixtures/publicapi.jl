using NLLSModel, Test

# rules.md 1 and 10: a name is public (exported, documented, in the API reference) or internal (none of these).
# Adding or dropping an export is a visible change to this list.
const PUBLIC_NAMES = sort([
    :AbstractAdaptiveResidual, :AbstractAdaptiveRobustifier, :AbstractCost, :AbstractResidual, :AbstractRobustifier,
    :ContaminatedGaussian, :GaussianPlusUniform, :GemanMcclure2oKernel, :GemanMcclureKernel, :Huber2oKernel, :HuberKernel,
    :Model, :NLLSModel, :NLLSNLSModel, :NLPLSModel, :NoRobust, :OptimizationRecommendation, :Scaled,
    :SimpleError2, :SimpleError3, :SimpleError4,
    :addcost!, :addvariable!, :computecost, :computecostgradhess, :computeresidual, :computeresjac,
    :cost, :flat_optimvars_nls, :generatemeasurement, :getvars, :islinear, :ndeps, :nres,
    :optimizerobustifier, :optimizerobustifier!, :optimvars, :optimvars_nls, :recommendoptimizer, :robustify, :robustifydcost, :robustifydkernel, :robustkernel,
    :subproblem, :subproblem!, :syncvars!, :varindices, :withlogvolume,
])

# Removed, not deprecated (rules.md 9)
const REMOVED_NAMES = [:addresidual!, :isadditive, :optimizekernel]

# Internal: still defined, no longer exported
const INTERNAL_NAMES = [
    :partition_schur_problem, :clamped_count, :reset_clamped_count!, :nresiduals, :nls_callbacks, :islinearkernel,
    :MAX_STATIC_VAR, :MAX_STATIC_JAC, :MAX_STATIC_ROWS,
]

# `Base.Docs.hasdoc` is not in Julia 1.10. A docstring is stored in the `meta` of the module that wrote it, under the
# binding of the owning module (`syncvars!` belongs to OptimizationVariableInterface).
function hasdocstring(mod::Module, name::Symbol)
    candidates = [Base.Docs.Binding(mod, name)]
    obj = getfield(mod, name)
    obj isa Function && push!(candidates, Base.Docs.Binding(parentmodule(obj), nameof(obj)))
    return any(b -> haskey(Base.Docs.meta(mod), b) || haskey(Base.Docs.meta(b.mod), b), candidates)
end

# The names listed in the `@docs` blocks of the API reference
function apireference_names(path)
    names, indocs = Symbol[], false
    for line in eachline(path)
        s = strip(line)
        if startswith(s, "```@docs")
            indocs = true
        elseif s == "```"
            indocs = false
        elseif indocs && !isempty(s)
            push!(names, Symbol(s))
        end
    end
    return sort(names)
end
