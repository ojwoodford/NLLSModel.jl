# NLLSModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ojwoodford.github.io/NLLSModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/NLLSModels.jl/dev/)
[![Build Status](https://github.com/ojwoodford/NLLSModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/NLLSModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/NLLSModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/NLLSModels.jl)

Nonlinear Least Squares (NLLS) and factor-graph modeling in Julia.

## Features

- **Factor-Graph Problem Representation**: Build bipartite variable-cost graphs with `NLLSProblem`, `addvariable!`, and `addcost!`.
- **Heterogeneous Variables & Manifolds**: Mix Euclidean `Float64`, `SVector`, `Vector`, and custom, non-Euclidean manifold types (via `OptimizationVariableInterface.jl`) in a single problem.
- **Automatic & Analytic Derivatives**: Uses `ForwardDiff.jl` automatically on `computeresidual`, or user-defined analytic Jacobians via `computeresjac`.
- **Robust M-Estimators**: Built-in `HuberKernel`, `GemanMcclureKernel`, `Scaled`, and adaptive `ContaminatedGaussian`.
- **Block-Dense & Block-Sparse Linear Algebra**: Uses `BlockDenseMatrix` and `BlockSparseMatrix` from `BlockMatrices.jl` for fast, zero-allocation Hessian assembly.
- **Optimizer-Agnostic Problem Interface**: Extract `(f, fdf!, x0, H) = optimvars(problem, unfixed, numthreads)` for seamless integration with many second order optimizers, such as `SecondOrderOptimization.jl` or `Optim.jl`.
- **NLPModels.jl Ecosystem**: Load `NLPModels` to wrap any `NLLSProblem` as an `AbstractNLPModel` via `NLLSModel(problem)` and solve with `JSOSolvers.jl` (`trunk`, `tron`), `Ipopt.jl`, or `Optim.jl`.

## Quick Start

```julia
using NLLSModels, SecondOrderOptimization, Static, StaticArrays

struct OffsetResidual <: AbstractResidual
    target::SVector{2, Float64}
    varind::Int
end
NLLSModels.ndeps(::OffsetResidual) = static(1)
NLLSModels.nres(::OffsetResidual) = static(2)
NLLSModels.varindices(r::OffsetResidual) = SVector(r.varind)
NLLSModels.getvars(r::OffsetResidual, vars::Vector) = (vars[r.varind]::SVector{2, Float64},)
NLLSModels.computeresidual(r::OffsetResidual, x) = x - r.target
Base.eltype(::OffsetResidual) = Float64

problem = NLLSProblem(SVector{2, Float64}, OffsetResidual)
v = addvariable!(problem, SVector(0.0, 0.0))
addcost!(problem, OffsetResidual(SVector(3.0, -2.0), v))

f, fdf!, x0, H = optimvars(problem)
x_opt, result = optimize(f, fdf!, x0, OptimizationOptions(iterator = :levenbergmarquardt); H = H)
```

### Solving with NLPModels.jl and JSOSolvers

```julia
using NLLSModels, NLPModels, JSOSolvers

# Wrap the NLLSProblem as an AbstractNLPModel
nlp = NLLSModel(problem)

# Optimize using trust-region solver
stats = trunk(nlp)

# Synchronize the optimal variables back into the problem
syncvars!(nlp, stats.solution)
```
