# NLLSModel.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ojwoodford.github.io/NLLSModel.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/NLLSModel.jl/dev/)
[![Build Status](https://github.com/ojwoodford/NLLSModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/NLLSModel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/NLLSModel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/NLLSModel.jl)

Nonlinear Least Squares (NLLS) and factor-graph modeling in Julia.

## Features

- **Factor-Graph Problem Representation**: Build bipartite variable-cost graphs with `Model`, `addvariable!`, and `addcost!`.
- **Heterogeneous Variables & Manifolds**: Mix Euclidean `Float64`, `SVector`, `Vector`, and custom, non-Euclidean manifold types (via `OptimizationVariableInterface.jl`) in a single problem.
- **Automatic & Analytic Derivatives**: Uses `ForwardDiff.jl` automatically on `computeresidual`, or user-defined analytic Jacobians via `computeresjac`.
- **Robust M-Estimators**: Built-in `HuberKernel`, `GemanMcclureKernel`, `Scaled`, and adaptive `ContaminatedGaussian`.
- **Block-Dense & Block-Sparse Linear Systems**: Uses `DenseBlockLinearSystem` and `SparseBlockLinearSystem` from `BlockLinearSystems.jl` for fast, zero-allocation Hessian assembly.
- **Optimizer-Agnostic Problem Interface**: Extract `(f, fdf!, x0, H) = optimvars(problem, unfixed, numthreads)` for seamless integration with many second order optimizers, such as `SecondOrderOptimization.jl` or `Optim.jl`.
- **NLPModels.jl Ecosystem**: Load `NLPModels` to wrap any `Model` as an `AbstractNLPModel` via `NLPLSModel(problem)` and solve with `JSOSolvers.jl` (`trunk`, `tron`), `Ipopt.jl`, or `Optim.jl`.

## Quick Start

```julia
using NLLSModel, SecondOrderOptimization, Static, StaticArrays

struct OffsetResidual <: AbstractResidual
    target::SVector{2, Float64}
    varind::Int
end
NLLSModel.ndeps(::OffsetResidual) = static(1)
NLLSModel.nres(::OffsetResidual) = static(2)
NLLSModel.varindices(r::OffsetResidual) = SVector(r.varind)
NLLSModel.getvars(r::OffsetResidual, vars::Vector) = (vars[r.varind]::SVector{2, Float64},)
NLLSModel.computeresidual(r::OffsetResidual, x) = x - r.target
Base.eltype(::OffsetResidual) = Float64

problem = Model(SVector{2, Float64}, OffsetResidual)
v = addvariable!(problem, SVector(0.0, 0.0))
addcost!(problem, OffsetResidual(SVector(3.0, -2.0), v))

f, fdf!, x0, H = optimvars(problem)
x_opt, result = optimize(f, fdf!, x0, OptimizationOptions(iterator = :levenbergmarquardt); H = H)
```

### Solving with NLPModels.jl and JSOSolvers

```julia
using NLLSModel, NLPModels, JSOSolvers

# Wrap the Model as an AbstractNLPModel
nlp = NLPLSModel(problem)

# Optimize using trust-region solver
stats = trunk(nlp)

# Synchronize the optimal variables back into the problem
syncvars!(nlp, stats.solution)
```

## Performance

Everything computed for one cost is single-threaded, by design: all threading is in the outer loop over costs, so no
BLAS call or threaded loop may run inside a cost. A cost with a very large single residual therefore does not get
threaded, cache-blocked BLAS; if that matters, supply `computecostgradhess` by hand.

For each squared cost, the gradient `g = dc * J'r` and Hessian `H = dc * J'J + c2 * (J'r)(J'r)'` are formed by one of
three kernels, chosen at compile time from the types involved (`nres` and `nvars` returning a `StaticInt` is what makes
a size static):

| # | Kernel | Used when |
|---|---|---|
| 1 | Full static multiply (`SMatrix`, unrolled) | every unfixed variable has a static size, totalling `n <= MAX_STATIC_VAR` (64); `nres` is a `StaticInt` `M`; and `M * n <= MAX_STATIC_JAC` (512) |
| 2 | Static-column kernel | otherwise, when `nres` is a `StaticInt` `M <= MAX_STATIC_ROWS` (16) and the Jacobian is a `Matrix` of `Float32`/`Float64` |
| 3 | `@turbo` kernel (plain loops for other array types) | otherwise |

Costs not on the full static path use ForwardDiff with run-time sized buffers, cached per task. `MAX_STATIC_VAR`,
`MAX_STATIC_JAC` and `MAX_STATIC_ROWS` are starting values that have not been tuned; they are constants in `src/NLLSModel.jl`.
A scalar residual (`nres == 1`) with a run-time sized variable uses an outer-product kernel.

Rules for your own `computeresidual`, `computeresjac` and `computecostgradhess`, because residuals are evaluated on
several threads at once:

- arrays they return must not be shared with other residuals;
- the code must itself be single-threaded (no BLAS on large matrices, no `Threads.@threads`). If it needs BLAS, set
  `BLAS.set_num_threads(1)` in the application.

NLLSModel depends directly on `LoopVectorization.jl` (for the `@turbo` kernel), alongside `BlockLinearSystems.jl`,
`FastNorm.jl`, `ForwardDiff.jl`, `Polyester.jl`, `OptimizationVariableInterface.jl` and `SIRBSDispatch.jl`.

