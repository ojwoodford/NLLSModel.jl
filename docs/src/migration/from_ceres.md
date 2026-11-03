# Migrating from Ceres Solver (C++)

[Ceres Solver](http://ceres-solver.org/) is an open-source C++ library for modeling and solving large nonlinear least squares problems.

Here is a direct mapping of Ceres Solver concepts to `NLLSModels.jl`:

## Concept Translation Table

| Ceres Solver (C++) | NLLSModels.jl (Julia) | Notes |
| :--- | :--- | :--- |
| `ceres::Problem` | [`NLLSProblem`](@ref) | Container for variable blocks and cost/residual blocks |
| `ceres::CostFunction` | [`AbstractResidual`](@ref) / [`AbstractCost`](@ref) | Defines residual vector & derivatives |
| `ceres::LossFunction` | [`AbstractRobustifier`](@ref) | Robust M-estimator kernel ([`HuberKernel`](@ref), [`Huber2oKernel`](@ref), [`GemanMcclure2oKernel`](@ref), etc.) |
| Parameter Block (`double*`) | Variable (`Float64`, `SVector`, `Vector`, custom struct) | Supports both immutable stack types and mutable arrays |
| `ceres::Manifold` | `OptimizationVariableInterface.nvars` & `OptimizationVariableInterface.update` | Tangent-space manifold retraction |
| `ceres::Solver::Options` | `OptimizationOptions` (`NLLSOptions`) | Iterator, linear solver, tolerances, thread count |
| `ceres::Solver::Summary` | `OptimizationResult` (`NLLSResult`) | Costs, timings, iteration/evaluation counts, allocations |
| `ceres::Solve(options, &problem, &summary)` | [`optimvars(problem)`](@ref optimvars) + `optimize` (or `NLLSsolver.optimize!`) | Core solve call |

## Ceres vs NLLSModels Code Comparison

### Ceres Solver (C++):
```cpp
struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

ceres::Problem problem;
double x = 0.5;
problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor),
    new ceres::HuberLoss(1.0),
    &x);

ceres::Solver::Options options;
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
```

### NLLSModels.jl (Julia):
```julia
using NLLSModels
using Static, StaticArrays

struct CostResidual <: AbstractResidual
    varind::Int
end
NLLSModels.ndeps(::CostResidual) = static(1)
NLLSModels.nres(::CostResidual) = static(1)
NLLSModels.varindices(r::CostResidual) = SVector(r.varind)
NLLSModels.getvars(r::CostResidual, vars::Vector) = (vars[r.varind]::Float64,)
NLLSModels.computeresidual(::CostResidual, x) = SVector(10.0 - x)
NLLSModels.robustkernel(::CostResidual) = Huber2oKernel(1.0)
Base.eltype(::CostResidual) = Float64

problem = NLLSProblem(Float64, CostResidual)
idx = addvariable!(problem, 0.5)
addcost!(problem, CostResidual(idx))

f, fdf!, x0, H = optimvars(problem)
options = OptimizationOptions(iterator = :levenbergmarquardt)
x_opt, result = optimize(f, fdf!, x0, options; H = H)
```
