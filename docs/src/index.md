# NLLSModel.jl

**Nonlinear Least Squares & Factor Graph Modeling in Julia**

`NLLSModel.jl` is a high-performance framework for formulating nonlinear least squares and general factor-graph optimization problems:

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
- **Fast Block Linear Systems**: Custom `DenseBlockLinearSystem` and `SparseBlockLinearSystem` (from `BlockLinearSystems.jl`) tailored for sparse $LDL^T$, Cholesky, and QR factorizations.
- **Automatic Differentiation**: Evaluates residual Jacobians and cost gradients/Hessians automatically via `ForwardDiff.jl`, or uses user-supplied analytic derivatives via [`computeresjac`](@ref).
- **Robust M-Estimators**: First- and second-order robust kernels plus adaptive robustifiers ([`ContaminatedGaussian`](@ref)).
- **One-Call Solving**: `optimize!(model, options)` (with `using SecondOrderOptimization`) solves the model and writes the solution back.
- **Solver-Agnostic Extraction**: For advanced use, [`optimvars`](@ref) returns `(f, fdf!, x0, H)` for custom outer loops or other optimizers.
- **Multi-threaded Evaluation**: Automatic thread-parallel residual and Hessian evaluation via `Polyester.jl`.

## Quick Start

```julia
using NLLSModel, SecondOrderOptimization
using Static, StaticArrays

# 1. Define a custom residual struct
struct PointDistanceResidual <: AbstractResidual
    target::SVector{2, Float64}
    varind::Int
end

NLLSModel.ndeps(::PointDistanceResidual) = static(1)
NLLSModel.nres(::PointDistanceResidual) = static(2)
NLLSModel.varindices(res::PointDistanceResidual) = SVector(res.varind)
NLLSModel.getvars(res::PointDistanceResidual, vars::Vector) = (vars[res.varind]::SVector{2, Float64},)
NLLSModel.computeresidual(res::PointDistanceResidual, x) = x - res.target
Base.eltype(::PointDistanceResidual) = Float64

# 2. Build the model
problem = Model(SVector{2, Float64}, PointDistanceResidual)
idx = addvariable!(problem, SVector(0.0, 0.0))
addcost!(problem, PointDistanceResidual(SVector(3.0, 4.0), idx))

# 3. Solve. The solution is written back into the model
result = optimize!(problem, OptimizationOptions(iterator = :levenbergmarquardt))
println("Estimated variable: ", problem.variables[idx])
println("Final cost: ", result.bestcost)
```

`optimize!(model, options)` is defined by a package extension: it needs `using SecondOrderOptimization` (NLLSModel already
depends on `BlockLinearSystems`). `optimize(model, options)` returns `(model_copy, result)` and leaves `model` unchanged.
The keywords `reduction` and `storage` choose the linear-system layout; see [Linear Solvers & Sparsity](@ref). Every way of
building or optimizing a model takes the same keywords (see [`optimvars`](@ref)).

## Tall formulation

`optimize!(model, options; formulation = :tall)` solves the same model from the residual vector and the Jacobian instead of
the normal equations. Each step is a QR factorization of the Jacobian (dense, or SuiteSparse's sparse QR), so `JᵀJ` is
never formed and the conditioning is that of `J`, not its square. Use it when the model is ill conditioned (the normal
equations stall above the minimum).

```julia
result = optimize!(problem, OptimizationOptions(iterator = :levenbergmarquardt); formulation = :tall)
```

In this first version:

- `storage` chooses a dense or a sparse Jacobian (`:auto` decides from its fill, and never chooses a dense one above
  `max_dense_bytes`, default 256 MB). A dense one forced above the cap is an `ArgumentError` giving its size, before
  anything is allocated;
- any iterator except gradient descent works (`:levenbergmarquardt`, `:levenbergmarquardt_dtd`, `:dogleg`, `:newton`), with
  `linsolver = :auto` or `:qr`; `:gradientdescent` and the other solvers (for example `:cholesky`) throw the optimizer's `ArgumentError`;
- `reduction = :eliminate` is an `ArgumentError` (use `:auto` or `:none`); `min_work` is the floor of the residual and
  Jacobian fill, in its own work estimate, with its own default;
- every cost must be a non-adaptive `AbstractResidual`, and no adaptive kernel variable may be unfixed. Otherwise the error
  from [`optimvars_nls`](@ref) is rethrown with the prefix `formulation = :tall:`;
- there is no robust-mode keyword: each cost's kernel decides how its rows enter (see the robust-kernel tutorial), so
  choosing `HuberKernel` or `Huber2oKernel` chooses first- or second-order treatment.

`formulation` is a keyword of `optimize!` and `optimize` only. It is not an `OptimizationOptions` field and is not inferred from
`linsolver`; a caller of `optimvars` or `optimvars_nls` chooses by which one they call.

## Advanced Use: `optimvars`

Use [`optimvars`](@ref) when you need the pieces yourself: a custom outer loop, a reused `OptimizationWorkspace`, the NLPModels
wrapper, or another optimizer.

```julia
f, fdf!, x0, H = optimvars(problem)
result = optimize!(x0, HessianObjective(f, fdf!; H), OptimizationOptions(iterator = :levenbergmarquardt))
syncvars!(problem, fdf!, x0)
```

A single variable of static size is immutable: call `x, result = optimize(x0, HessianObjective(f, fdf!; H), options)` and
`syncvars!(problem, fdf!, x)`.
