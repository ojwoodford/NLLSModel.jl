# NLLSModels.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/NLLSModels.jl/dev/)
[![Build Status](https://github.com/ojwoodford/NLLSModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/NLLSModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/NLLSModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/NLLSModels.jl)

Nonlinear Least Squares (NLLS) and factor-graph modeling in Julia.

## Features

- **Factor-Graph Problem Representation**: Build bipartite variable-cost graphs with `NLLSProblem`, `addvariable!`, and `addcost!`.
- **Heterogeneous Variables & Manifolds**: Mix `Float64`, `SVector`, `Vector`, `ZeroToInfScalar`, `ZeroToOneScalar`, and custom manifold types (via `OptimizationVariableInterface.jl`) in a single problem.
- **Automatic & Analytic Derivatives**: Uses `ForwardDiff.jl` automatically on `computeresidual`, or user-defined analytic Jacobians via `computeresjac`.
- **Robust M-Estimators**: Built-in `HuberKernel`, `Huber2oKernel`, `GemanMcclureKernel`, `GemanMcclure2oKernel`, `Scaled`, and adaptive `ContaminatedGaussian`.
- **Block-Dense & Block-Sparse Linear Algebra**: Uses `BlockDenseMatrix` and `BlockSparseMatrix` from `BlockMatrices.jl` for fast, zero-allocation Hessian assembly.
- **Solver-Agnostic Problem Interface**: Extract `(f, fdf!, x0, H) = optimvars(problem, unfixed, numthreads)` for seamless integration with `SecondOrderOptimization.jl` or `NLLSsolver.jl`.

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
