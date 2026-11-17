# API Reference

```@docs
NLLSModel
```

## Problem Representation

```@docs
Model
addvariable!
addcost!
subproblem
subproblem!
cost
optimvars
optimvars_nls
flat_optimvars_nls
islinear
```

## Solving a Model

With `using SecondOrderOptimization`, a package extension adds two methods to its functions:

- `optimize!(model::Model, options = OptimizationOptions(); formulation = :normal, callback, workspace, unfixed, numthreads, reduction = :auto, storage = :auto, max_dense_bytes, min_work, min_work_cost) -> result` solves the model and writes the solution into `model.variables`. `formulation = :tall` solves from the residual and the Jacobian (see the home page).
- `optimize(model::Model, options = OptimizationOptions(); kwargs...) -> (model_copy, result)` does the same on a copy.

The keywords after `workspace` are the set every entry point takes, described under [`optimvars`](@ref) and in
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
NLLSNLSModel
syncvars!
```
