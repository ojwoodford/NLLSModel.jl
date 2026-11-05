# NLLSModels.jl

**Nonlinear Least Squares & Factor Graph Modeling in Julia**

`NLLSModels.jl` is a high-performance framework for formulating nonlinear least squares and general factor-graph optimization problems:

```math
\min_{\mathbf{x}} \sum_{i} \frac{1}{2} \rho_i \left( \| r_i(\mathbf{x}_{S_i}) \|^2 \right) + \sum_{j} c_j(\mathbf{x}_{S_j})
```

where:
- $\mathbf{x} = \{ \mathbf{x}_1, \dots, \mathbf{x}_N \}$ are variable blocks (scalars, static vectors, dynamic vectors, or custom manifolds).
- $r_i$ is the $i$-th residual function depending on a subset of variables $\mathbf{x}_{S_i}$.
- $\rho_i$ is an optional robust loss kernel ([`HuberKernel`](@ref), [`Huber2oKernel`](@ref), [`GemanMcclureKernel`](@ref), [`GemanMcclure2oKernel`](@ref), [`ContaminatedGaussian`](@ref), [`Scaled`](@ref)).
- $c_j$ is an optional general scalar cost block ([`AbstractCost`](@ref)).

## Features

- **Heterogeneous Variable & Cost Blocks**: Store variable blocks and cost types with zero type-instability.
- **Fast Block Linear Algebra**: Custom `BlockDenseMatrix` and `BlockSparseMatrix` (from `BlockMatrices.jl`) tailored for sparse $LDL^T$, Cholesky, and QR factorizations.
- **Automatic Differentiation**: Evaluates residual Jacobians and cost gradients/Hessians automatically via `ForwardDiff.jl`, or uses user-supplied analytic derivatives via [`computeresjac`](@ref).
- **Robust M-Estimators**: First- and second-order robust kernels plus adaptive robustifiers ([`ContaminatedGaussian`](@ref)).
- **Solver-Agnostic Extraction**: Use [`optimvars`](@ref) to obtain `(f, fdf!, x0, H)` for direct use with `SecondOrderOptimization.jl` or `NLLSsolver.jl`.
- **Multi-threaded Evaluation**: Automatic thread-parallel residual and Hessian evaluation via `Polyester.jl`.

## Quick Start

```julia
using NLLSModels, SecondOrderOptimization
using Static, StaticArrays

# 1. Define a custom residual struct
struct PointDistanceResidual <: AbstractResidual
    target::SVector{2, Float64}
    varind::Int
end

NLLSModels.ndeps(::PointDistanceResidual) = static(1)
NLLSModels.nres(::PointDistanceResidual) = static(2)
NLLSModels.varindices(res::PointDistanceResidual) = SVector(res.varind)
NLLSModels.getvars(res::PointDistanceResidual, vars::Vector) = (vars[res.varind]::SVector{2, Float64},)
NLLSModels.computeresidual(res::PointDistanceResidual, x) = x - res.target
Base.eltype(::PointDistanceResidual) = Float64

# 2. Build the problem
problem = NLLSProblem(SVector{2, Float64}, PointDistanceResidual)
idx = addvariable!(problem, SVector(0.0, 0.0))
addcost!(problem, PointDistanceResidual(SVector(3.0, 4.0), idx))

# 3. Extract optimization callables and solve
f, fdf!, x0, H = optimvars(problem)
x_opt, result = optimize(f, fdf!, x0, OptimizationOptions(iterator = :levenbergmarquardt); H = H)
println("Estimated variable: ", x_opt)
println("Final cost: ", result.bestcost)
```
