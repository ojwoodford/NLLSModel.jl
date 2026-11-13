# API Reference

## Problem Representation

```@docs
Model
addvariable!
addcost!
addresidual!
subproblem
subproblem!
cost
optimvars
partition_schur_problem
optimvars_nls
nls_callbacks
flat_optimvars_nls
islinear
```

## Solving a Model

With `using SecondOrderOptimization`, a package extension adds two methods to its functions:

- `optimize!(model::Model, options = OptimizationOptions(); formulation = :normal, unfixed, numthreads, callback, workspace, reduction = :auto, reduced = :auto, min_work, min_work_cost, min_work_per_thread, max_dense_bytes) -> result` solves the model and writes the solution into `model.variables`. `formulation = :tall` solves from the residual and a dense Jacobian (see the home page).
- `optimize(model::Model, options = OptimizationOptions(); kwargs...) -> (model_copy, result)` does the same on a copy.

`reduction` (`:auto`, `:none`, `:eliminate`) and `reduced` (`:auto`, `:dense`, `:sparse`) are described under [`optimvars`](@ref) and in
[Linear Solvers & Sparsity](@ref). For custom outer loops use [`optimvars`](@ref) and `HessianObjective` directly.

## Cost & Residual Interface

```@docs
AbstractCost
AbstractResidual
AbstractAdaptiveResidual
ndeps
nres
varindices
getvars
computeresidual
computeresjac
computecost
computecostgradhess
SimpleError2
SimpleError3
SimpleError4
generatemeasurement
```

## Static-Size Limits

```@docs
NLLSModel.MAX_STATIC_VAR
NLLSModel.MAX_STATIC_JAC
NLLSModel.MAX_STATIC_ROWS
```

## Robust Cost Functions

```@docs
AbstractRobustifier
AbstractAdaptiveRobustifier
robustkernel
robustify
robustifydcost
robustifydkernel
NoRobust
Scaled
HuberKernel
Huber2oKernel
GemanMcclureKernel
GemanMcclure2oKernel
ContaminatedGaussian
```

## NLPModels Integration

```@docs
NLPLSModel
syncvars!
```
