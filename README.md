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
- **One-Call Solving**: `optimize!(model, options)` solves a `Model` with `SecondOrderOptimization.jl` and writes the solution back into it (load `SecondOrderOptimization`).
- **Optimizer-Agnostic Problem Interface**: For advanced use, extract `(f, fdf!, x0, H) = optimvars(problem, unfixed, numthreads)` for integration with other second order optimizers, such as `Optim.jl`.
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

result = optimize!(problem, OptimizationOptions(iterator = :levenbergmarquardt))
problem.variables[v] # The solution
```

`optimize!(model, options)` is provided by a package extension, so it needs `using SecondOrderOptimization` (`BlockLinearSystems`
is already a dependency of NLLSModel). `optimize(model, options)` leaves `model` alone and returns `(model_copy, result)`.
Keywords: `unfixed` (an index, a type, a mask or a list of indices), `numthreads`, `callback`, `workspace`, and the layout
choices `reduction` (`:auto`, `:none`, `:eliminate`: whether eliminable variables are eliminated by a Schur complement) and
`reduced` (`:auto`, `:dense`, `:sparse`: storage of the reduced system, ignored when nothing is eliminated).

### Advanced use: `optimvars`

`optimvars` is needed for a custom outer loop, for reusing an `OptimizationWorkspace` across solves with your own objective
handling, or to feed the NLPModels wrapper or another optimizer. It returns the cost, the in-place gradient/Hessian
evaluator, the starting variables and the Hessian prototype, which you wrap in a `HessianObjective`:

```julia
f, fdf!, x0, H = optimvars(problem)
result = optimize!(x0, HessianObjective(f, fdf!; H), OptimizationOptions(iterator = :levenbergmarquardt))
syncvars!(problem, fdf!, x0) # Write the solution back
```

`SecondOrderOptimization.optimize!` needs a mutable `x0`; a single variable of static size is immutable, so use
`x, result = optimize(x0, HessianObjective(f, fdf!; H), options)` and `syncvars!(problem, fdf!, x)` there.

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
the kernels below, chosen from the types involved (`nres` and `nvars` returning a `StaticInt` is what makes a size
static). Rows 2 to 4 are tried in order; the first that applies runs:

| # | Kernel | Used when |
|---|---|---|
| 1 | Full static multiply (`SMatrix`, unrolled) | every unfixed variable has a static size, totalling `n <= MAX_STATIC_VAR` (64); `nres` is a `StaticInt` `M`; and `M * n <= MAX_STATIC_JAC` (128) |
| 2 | Static-column kernel | otherwise, when `nres` is a `StaticInt` `M <= MAX_STATIC_ROWS` (16) and the Jacobian is a `Matrix` of `Float32`/`Float64` |
| 3 | Plain loops, single column | the Jacobian has one column (`n == 1`), whatever the types: a single long dot product, which plain loops do 2 to 4 times faster than `@turbo` for 100 to 1000 rows |
| 4 | `@turbo` kernel | the Jacobian is a `Matrix` (or unit-range view) and the residual a `Vector` (or unit-range view) of the same `Float32`/`Float64` type, which is what dynamic autodiff produces, and a hand-written `(Vector, Matrix)` |
| 5 | Plain loops (`@fastmath @inbounds @simd`) | anything `@turbo` cannot address: an `SVector` longer than `MAX_STATIC_ROWS`, or an `MVector` of any length, with a `Matrix` Jacobian (hand-written `computeresjac` only), an `Adjoint`, a strided view, mixed element types. Intended, not a gap: nothing is copied |

On an Apple M1 Pro, `@turbo` was 1.2 to 2 times faster than the plain loops over most of the grid of 1 to 1000 rows and 1
to 250 columns, and 22 to 34% faster on a linearization the kernels dominate; see `benchmark/results/`. The plain loops
carry a `NaN` or `Inf` in the Jacobian or residual through to the gradient and Hessian, as the optimizer's termination
tests need.

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

