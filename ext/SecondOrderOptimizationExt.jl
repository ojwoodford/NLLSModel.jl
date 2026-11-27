module SecondOrderOptimizationExt

using NLLSModel
using SecondOrderOptimization
using OptimizationVariableInterface
using Static
using StaticArrays: SVector
using PrecompileTools: @setup_workload, @compile_workload
import NLLSModel: MIN_WORK_PER_THREAD_HESSIAN, MIN_WORK_PER_THREAD_COST, MIN_WORK_PER_THREAD_JACOBIAN, MAX_DENSE_BYTES, KEYWORDS_DOC, JACOBIAN_FORMULATION_DOC,
    OptimizationRecommendation

# Hands `resjac!` to the optimizer with `prezeroed = true`: the structural zeros of `J` are already in place (the optimizer
# allocates `J` as zeros and, from SecondOrderOptimization's Z1, neither re-zeroes it nor writes anything but this evaluator's
# entries), and the evaluator writes every entry that can be non-zero on every call, so it need not fill `J` first.
# A struct rather than a closure, so the objective's type is concrete.
struct JacobianResJac{RJ}
    resjac!::RJ
end
(w::JacobianResJac)(r, J, x) = w.resjac!(r, J, x; prezeroed=true)

# What a solve of a model is built from: the objective, the cost functor (which knows the unfixed variables), the start
# point and the functor that writes a solution back. A `ModelWorkspace` keeps one between calls.
struct ModelSetup{O, F, X, S}
    obj::O
    f::F
    x0::X
    sync::S
end

# The Hessian formulation: `optimvars`, wrapped in a `HessianObjective`
function hessiansetup(model::Model, keywords::NamedTuple)
    f, fdf!, x0, H = optimvars(model; keywords...)
    return ModelSetup(HessianObjective(f, fdf!; H), f, x0, fdf!)
end

# The Jacobian formulation: the residual vector and the Jacobian, solved by the optimizer from the Jacobian prototype that
# `optimvars_nls` returns (its type decides the linear system; the optimizer validates it against `linsolver`). A dense
# Jacobian forced above `max_dense_bytes` is refused by `optimvars_nls` before it is allocated.
function jacobiansetup(model::Model, keywords::NamedTuple)
    f, _, resjac!, x0, J = try
        optimvars_nls(model; keywords...)
    catch err
        err isa ArgumentError ? throw(ArgumentError("formulation = :jacobian: " * err.msg)) : rethrow()
    end
    return ModelSetup(JacobianObjective(f, JacobianResJac(resjac!), J), f, x0, resjac!)
end

# The start point of a stored setup, read from the model's variables as they are now: a single variable is the model's
# own; several are copied into the setup's vector, unless it is the model's variable vector itself
startpoint!(x0, f::NLLSModel.SingleVarCostFunc, model::Model) = @inbounds model.variables[f.unfixed]::typeof(x0)
function startpoint!(x0, f::NLLSModel.MultiVarCostFunc, model::Model)
    inds = f.unfixed_inds
    isnothing(inds) && return x0
    @inbounds for k in eachindex(inds)
        x0[k] = model.variables[inds[k]]
    end
    return x0
end

# Solves from a setup and writes the solution back with `syncvars!`, which knows the block order and the Schur partition.
# A single variable of static size is immutable, so it goes through `optimize` (which returns the solution); everything
# else is updated in place by `optimize!`. `callback` is a type parameter here and in the callers below, so that these
# methods are specialized on it (a function passed through is otherwise not), and a cached call dispatches nothing.
function runsetup!(model::Model, s::ModelSetup, x0, options, callback::CB, workspace) where CB
    if ismutablevar(x0)
        result = SecondOrderOptimization.optimize!(x0, s.obj, options; callback, workspace)
        syncvars!(model, s.sync, x0)
    else
        x, result = SecondOrderOptimization.optimize(x0, s.obj, options; callback, workspace)
        syncvars!(model, s.sync, x)
    end
    return result::OptimizationResult
end
runsetup!(model::Model, s::ModelSetup, options, callback::CB, workspace) where CB = runsetup!(model, s, s.x0, options, callback, workspace)

function checkkeywords(formulation::Symbol, reduction::Symbol, storage::Symbol, precision::Type)
    formulation in (:hessian, :jacobian) || throw(ArgumentError("Unknown formulation $(repr(formulation)); expected :hessian or :jacobian."))
    NLLSModel.check_reduction(reduction)
    NLLSModel.check_storage(storage)
    NLLSModel.check_precision(precision)
    return nothing
end

# The keywords a setup is keyed on, with the numbers as Ints (a StaticInt and an Int of the same value are one key)
keykeywords(formulation::Symbol, k::NamedTuple) = (; formulation, k.unfixed, numthreads = Int(dynamic(k.numthreads)), k.reduction,
    k.storage, max_dense_bytes = Int(k.max_dense_bytes), min_work = Int(k.min_work), min_work_cost = Int(k.min_work_cost), k.precision)

newsetup(model::Model, formulation::Symbol, keywords::NamedTuple) =
    formulation === :jacobian ? jacobiansetup(model, keywords) : hessiansetup(model, keywords)

# A solve with a ModelWorkspace: the stored setup when its key matches, otherwise a new one, stored in the workspace
function optimize_cached!(model::Model, options, formulation::Symbol, keywords::NamedTuple, callback::CB, ws::ModelWorkspace) where CB
    if NLLSModel.keymatches(ws.key, model, keykeywords(formulation, keywords))
        s = something(ws.setup)
        return runsetup!(model, s, startpoint!(s.x0, s.f, model), options, callback, ws.optimizer)
    end
    return optimize_newsetup!(model, options, formulation, keywords, callback, ws)
end
@noinline function optimize_newsetup!(model::Model, options, formulation::Symbol, keywords::NamedTuple, callback::CB, ws::ModelWorkspace) where CB
    s = newsetup(model, formulation, keywords)
    storesetup!(ws, NLLSModel.setupkey(model, keykeywords(formulation, keywords)), s)
    return runsetup!(model, s, options, callback, ws.optimizer)
end

function storesetup!(ws::ModelWorkspace{K, S}, key, setup) where {K, S}
    if key isa K && setup isa S
        ws.key = key
        ws.setup = setup
    else
        @warn "ModelWorkspace{$K, $S} can't hold the setup for this call, so it was emptied and every call with it builds the setup. Use `ModelWorkspace()` for a workspace that adapts to any call." maxlog=1
        empty!(ws)
    end
    return ws
end

"""
    optimize!(model::Model, options = OptimizationOptions(); formulation=:hessian, callback=nullcallback, workspace=nothing, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64) -> OptimizationResult

Solve `model` and write the solution back into `model.variables`. This is the normal way to solve a `Model`; it is defined when
`SecondOrderOptimization` is loaded (`using SecondOrderOptimization`).

The second method, `optimize!(model, rec)`, takes every choice from an `OptimizationRecommendation` made by
[`recommendoptimizer`](@ref); its docstring is below.

`formulation` chooses how the model is linearized:

- `:hessian` (default): the Hessian (for residuals, the normal equations). Calls [`optimvars`](@ref), wraps the result in a `HessianObjective`, runs
  `SecondOrderOptimization.optimize!` (or `optimize` when the variable to optimize is immutable, a single variable of static
  size) and writes the solution back with [`syncvars!`](@ref).
- `:jacobian`: the residual vector and the Jacobian (a `JacobianObjective`); the whole `JᵀJ` is never formed, unless
  `linsolver` is `:cholesky` or `:ldl` with a dense Jacobian and no elimination (the normal equations, formed from `J`).
  Calls [`optimvars_nls`](@ref) with the same keywords and hands its Jacobian prototype to the optimizer, which chooses the
  linear system from it; the table below. Each cost's kernel decides how its rows enter; there is no robust keyword. Every
  cost must be a non-adaptive `AbstractResidual`, with no unfixed adaptive kernel; errors from `optimvars_nls` are
  rethrown prefixed `formulation = :jacobian:`, and a dense Jacobian forced above `max_dense_bytes` is one of them, raised
  before it is allocated.

$JACOBIAN_FORMULATION_DOC

`formulation` is a keyword here only: it is not a field of `OptimizationOptions` and is not inferred from `options.linsolver`. Any
other value is an `ArgumentError`. `precision = Float32` builds the linear system (on either path, and its elimination) in
single precision, everything else in double; see the keywords below, and the "Single Precision" section of the
linear-solvers tutorial for when it pays. `callback` is passed to the optimizer. The default of `min_work` is the
formulation's ($MIN_WORK_PER_THREAD_HESSIAN for `:hessian`, $MIN_WORK_PER_THREAD_JACOBIAN for `:jacobian`).

`workspace` is one of:

- `nothing` (default): everything is built for this call.
- an `OptimizationWorkspace`: passed to the optimizer, which reuses its own state (its linear system, factorization and
  iterator state) when it fits. The model's setup (what `optimvars` or `optimvars_nls` builds) is built for each call.
- a [`ModelWorkspace`](@ref): the model's setup is reused too, while the model's structure and the keywords are those it
  was built for, so a repeated call with a typed one allocates nothing.

$KEYWORDS_DOC"""
function SecondOrderOptimization.optimize!(model::Model, options::OptimizationOptions=OptimizationOptions();
        formulation::Symbol=:hessian, callback=nullcallback, workspace=nothing,
        unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction::Symbol=:auto, storage::Symbol=:auto,
        max_dense_bytes::Integer=MAX_DENSE_BYTES,
        min_work::Integer=(formulation === :jacobian ? MIN_WORK_PER_THREAD_JACOBIAN : MIN_WORK_PER_THREAD_HESSIAN),
        min_work_cost::Integer=MIN_WORK_PER_THREAD_COST, precision::Type=Float64)
    checkkeywords(formulation, reduction, storage, precision)
    keywords = (; unfixed, numthreads, reduction, storage, max_dense_bytes, min_work, min_work_cost, precision)
    workspace isa ModelWorkspace && return optimize_cached!(model, options, formulation, keywords, callback, workspace)
    formulation === :jacobian && return runsetup!(model, jacobiansetup(model, keywords), options, callback, workspace)
    return runsetup!(model, hessiansetup(model, keywords), options, callback, workspace)
end

# The constructors of `ModelWorkspace` (its docstring is in NLLSModel): they need the optimizer's workspace
NLLSModel.ModelWorkspace() = ModelWorkspace{Any, Any, OptimizationWorkspace{Any, Any}}(nothing, nothing, OptimizationWorkspace())

function NLLSModel.ModelWorkspace(model::Model, options::OptimizationOptions=OptimizationOptions();
        formulation::Symbol=:hessian, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction::Symbol=:auto,
        storage::Symbol=:auto, max_dense_bytes::Integer=MAX_DENSE_BYTES,
        min_work::Integer=(formulation === :jacobian ? MIN_WORK_PER_THREAD_JACOBIAN : MIN_WORK_PER_THREAD_HESSIAN),
        min_work_cost::Integer=MIN_WORK_PER_THREAD_COST, precision::Type=Float64)
    checkkeywords(formulation, reduction, storage, precision)
    keywords = (; unfixed, numthreads, reduction, storage, max_dense_bytes, min_work, min_work_cost, precision)
    return typedworkspace(model, newsetup(model, formulation, keywords), options, NLLSModel.setupkey(model, keykeywords(formulation, keywords)))
end
function typedworkspace(model::Model, s::ModelSetup, options, key)
    optimizer = OptimizationWorkspace(s.x0, s.obj, options)
    return ModelWorkspace{typeof(key), typeof(s), typeof(optimizer)}(key, s, optimizer)
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
    optimize!(model::Model, rec::OptimizationRecommendation; callback=nullcallback, workspace=nothing, min_work, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64, options...) -> OptimizationResult

Solve `model` as the recommendation `rec` (from [`recommendoptimizer`](@ref)) says, and write the solution back into
`model.variables`. It builds `OptimizationOptions(; iterator = rec.iterator, linsolver = rec.linsolver, options...)` and
calls `optimize!(model, options; formulation, reduction, storage, unfixed, numthreads, max_dense_bytes)` with the
recommendation's values: exactly the call that displaying `rec` prints.

The remaining keywords are passed to the `OptimizationOptions` constructor (`maxiters`, `maxtime`, the tolerances, ...),
so an unknown one is an error there. A keyword the recommendation sets (`iterator`, `linsolver`, `formulation`,
`reduction`, `storage`, `unfixed`, `numthreads`, `max_dense_bytes`) is an `ArgumentError`, raised before anything is
built: to change one, use `optimize!(model, options; ...)`, starting from the call that `show` prints. `callback`,
`workspace`, `min_work` (default the formulation's), `min_work_cost` and `precision` are as for `optimize!(model, options)`:
the recommendation is made in `Float64` and recommends no precision, so `precision` is the caller's.

The options are built from the recommendation's values at run time, so the call dispatches once on their type. Setting
the options by hand, from the call that `show` prints, avoids it.
"""
function SecondOrderOptimization.optimize!(model::Model, rec::OptimizationRecommendation;
        callback=nullcallback, workspace=nothing,
        min_work::Integer=(rec.formulation === :jacobian ? MIN_WORK_PER_THREAD_JACOBIAN : MIN_WORK_PER_THREAD_HESSIAN),
        min_work_cost::Integer=MIN_WORK_PER_THREAD_COST, precision::Type=Float64, options...)
    # The one slurped keyword list in an entry point (rules.md 19): every keyword in it goes to the `OptimizationOptions`
    # constructor, whose keywords are spelled out, so a misspelt or stale keyword is still an error
    for name in keys(options)
        name in RECOMMENDATION_KEYWORDS && throw(ArgumentError("$name is set by the recommendation; to set it by hand, use optimize!(model, options; ...), starting from the call that displaying the recommendation prints."))
    end
    opts = OptimizationOptions(; iterator=rec.iterator, linsolver=rec.linsolver, options...)
    return SecondOrderOptimization.optimize!(model, opts; formulation=rec.formulation, reduction=rec.reduction,
        storage=rec.storage, unfixed=rec.unfixed, numthreads=rec.numthreads, max_dense_bytes=rec.max_dense_bytes,
        callback, workspace, min_work, min_work_cost, precision)
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

# optimizesingles! (NLLSsolver D6): each named variable optimized on its own, the others fixed. The variables are taken in
# runs of one type and size; a run's setup (its linear system and the optimizer's workspace) is built for its first
# variable and retargeted to each of the others, so a run builds one system, whatever its length.
function NLLSModel.optimizesingles!(model::Model{VT, CT}, options::OptimizationOptions, variables::Union{Type, AbstractVector{<:Integer}};
        precision::Type=Float64, max_dense_bytes::Integer=MAX_DENSE_BYTES) where {VT, CT}
    NLLSModel.check_precision(precision)
    indices = NLLSModel.singlesindices(model, variables)
    result = emptyresult(options)
    isempty(indices) && return result
    costsof = NLLSModel.costsbyvariable(model)
    sub = Model{VT, CT}(model.variables) # Shares the variables with the model; the costs of one variable at a time
    for run in NLLSModel.singlesruns(model, indices)
        result = optimizerun!(sub, model, options, run, costsof, precision, Int(max_dense_bytes), result)
    end
    return result
end

# The identity of `+` on results: what a call that optimizes no variable returns
emptyresult(options::OptimizationOptions) = OptimizationResult(0.0, 0.0, UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
    0, 0, 0, 0, 0, dynamic(options.iterator), 0, 0, 0, 0)

# One run: the setup is built for the run's first variable, on its costs (selected again in the loop, so that the loop
# holds for any run)
function optimizerun!(sub::Model, model::Model, options, run, costsof, precision, max_dense_bytes, result)
    first_ = first(run)
    subproblem!(sub, model, NLLSModel.costsof(costsof, first_))
    f, fdf!, x0, H = optimvars(sub; unfixed=first_, numthreads=StaticInt(1), max_dense_bytes, precision)
    workspace = OptimizationWorkspace(x0, HessianObjective(f, fdf!; H), options)
    return optimizerun!(sub, model, options, run, costsof, f, fdf!, x0, H, workspace, result)
end
function optimizerun!(sub::Model, model::Model, options, run, costsof, f, fdf!, x0::X, H, workspace, result) where X
    for i in run
        subproblem!(sub, model, NLLSModel.costsof(costsof, i))
        fi, fdfi! = NLLSModel.retarget(f, sub, i), NLLSModel.retarget(fdf!, i)
        obj = HessianObjective(fi, fdfi!; H)
        xi = @inbounds(sub.variables[i])::X
        if ismutablevar(xi)
            r = SecondOrderOptimization.optimize!(xi, obj, options; workspace)
            syncvars!(sub, fdfi!, xi)
        else
            x, r = SecondOrderOptimization.optimize(xi, obj, options; workspace)
            syncvars!(sub, fdfi!, x)
        end
        result += r
    end
    return result
end

# The precompile workload, last (improvements.md item 3)
include("SecondOrderOptimizationExtPrecompile.jl")

end
