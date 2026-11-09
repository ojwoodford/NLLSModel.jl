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
optimvars_nls
nls_callbacks
flat_optimvars_nls
```

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
syncvars!
```
