# API Reference

## Problem Representation

```@docs
NLLSProblem
addvariable!
addcost!
cost
costgradhess!
updatevarcostmap!
subproblem
subproblem!
VectorRepo
```

## Variables & Built-In Variable Types

```@docs
ZeroToInfScalar
ZeroToOneScalar
copyvars
isbitsvartype
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

## Optimization Interface

```@docs
optimvars
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

## Block Matrix Data Structures

```@docs
BlockDenseMatrix
BlockSparseMatrix
```
