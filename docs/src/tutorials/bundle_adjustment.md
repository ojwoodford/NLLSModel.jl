# Bundle Adjustment & Alternating Minimization

Bundle Adjustment (BA) jointly optimizes 3D landmark positions and camera poses to minimize reprojection errors:

```math
\min_{\{C_i\}, \{X_j\}} \sum_{i, j} \frac{1}{2} \rho\left(\| \pi(C_i, X_j) - z_{ij} \|^2\right)
```

## Modeling Bundle Adjustment with `SimpleError2`

Using `SVector` and [`SimpleError2`](@ref), you can define a bundle adjustment problem concisely by specializing [`NLLSModel.generatemeasurement`](@ref):

```julia
using NLLSModel
using Static, StaticArrays, LinearAlgebra

# Define projection of 3D point X by 6-DoF affine camera pose
function NLLSModel.generatemeasurement(pose::SVector{6, T}, X::SVector{3, U}) where {T, U}
    return SVector(
        dot(@inbounds(view(pose, SUnitRange(1, 3))), X),
        dot(@inbounds(view(pose, SUnitRange(4, 6))), X)
    )
end

const CamType = SVector{6, Float64}
const PtType  = SVector{3, Float64}
const ObsType = SimpleError2{2, Float64, CamType, PtType}

problem = Model(Union{CamType, PtType}, ObsType)
cam1 = addvariable!(problem, CamType(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
pt1  = addvariable!(problem, PtType(0.5, -0.2, 10.0))
addcost!(problem, SimpleError2{CamType, PtType}(SVector(0.5, -0.2), cam1, pt1))
```

## Joint Sparse Optimization vs Alternating Single-Variable Optimization

Because individual 3D landmarks do not interact directly with each other, you can either:
1. **Jointly optimize** all cameras and landmarks with `optimize!(problem, options)` (after `using SecondOrderOptimization`), which automatically uses `BlockSparseMatrix` and `LDLFactorizations.jl`, or eliminates the landmarks by a Schur complement when they are `iseliminable`.
2. **Optimize landmarks (or cameras) individually** in alternation using [`optimvars`](@ref) for a single variable index (or `NLLSsolver.optimizesingles!`), which solves a tiny $3 \times 3$ static linear system per 3D point with zero heap allocations:

```julia
using SecondOrderOptimization
options = OptimizationOptions(iterator = :levenbergmarquardt_dtd)

# Optimize a single 3D landmark (3x3 static system)
res_pt = optimize!(problem, options; unfixed = pt1)

# Or jointly optimize the full problem
res_joint = optimize!(problem, options)
```
