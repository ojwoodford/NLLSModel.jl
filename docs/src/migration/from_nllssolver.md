# Migrating from NLLSsolver.jl

`NLLSsolver.jl` has been refactored into two modular packages:
1. `SecondOrderOptimization.jl`: General-purpose, zero-allocation second-order nonlinear optimizer.
2. `NLLSModel.jl`: Nonlinear least squares factor-graph modeling, automatic differentiation, robust kernels, and block-sparse assembly.

`NLLSsolver.jl` remains available as a 100% backward-compatible umbrella package that re-exports both `SecondOrderOptimization` and `NLLSModel`.

## Key Changes

### 1. `Model` Struct Simplification & `CostStruct`
In previous versions of `NLLSsolver.jl`, `NLLSProblem` carried internal optimization scratch buffers (`varnext` and `varbest`), and stored costs in a `Dict`-backed `VectorRepo`.
In `NLLSModel.jl`, [`Model`](@ref) stores only the model data, using a zero-overhead tuple-backed `CostStruct` parameterized on the concrete cost types (`DataType`, `Union`, or `Tuple` of concrete cost types):
- `costs::CostStruct{CostTypes, CostVecs}`
- `variables::Vector{VarTypes}`
- `varcostmap::SparseMatrixCSC{Bool, Int}`
- `varcostmapvalid::Bool`

All optimization scratch buffers (`variables`, `varnext`, `varbest`) live inside `SecondOrderOptimization.OptimizationInternal` and `OptimizeCache` (`OptimizationWorkspace`). For backward compatibility with existing callbacks in `NLLSsolver.jl`, accessing `problem.varnext` or `problem.varbest` inside an `optimize!` callback continues to work transparently.

### 2. Decoupled Architecture & `optimvars`
`NLLSModel.jl` and `SecondOrderOptimization.jl` do not depend on each other:
- `NLLSModel.jl` provides [`optimvars`](@ref) returning `(f, fdf!, x0, H)` for any [`Model`](@ref).
- `SecondOrderOptimization.jl` provides `optimize(f, fdf!, x0, options; H=H)` along with `OptimizationOptions`, `OptimizationResult`, `OptimizationIterator`, `OptimizationInternal`, and `OptimizationWorkspace`.
- `NLLSsolver.jl` ties both packages together and provides `optimize!(problem, options)`, `optimizesingles!(problem, options)`, and the backward-compatible aliases (`NLLSOptions`, `NLLSResult`, `NLLSIterator`, `NLLSInternal`, `OptimizeCache`).

### 3. Separation of Concerns
If you want to optimize a smooth scalar objective `f(x)` with an in-place gradient/Hessian function `fdf!(g, H, x)` without constructing a `Model` factor graph, you can now use `SecondOrderOptimization.optimize(f, fdf!, x0, options)` directly.
