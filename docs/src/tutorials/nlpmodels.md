# Integration with NLPModels.jl

`NLLSModel.jl` provides seamless integration with the [JuliaSmoothOptimizers (JSO)](https://github.com/JuliaSmoothOptimizers) ecosystem via an `NLPModels.jl` package extension.

By loading `NLPModels`, any `NLLSModel` can be wrapped into an `AbstractNLPModel{Float64, Vector{Float64}}` via [`NLPLSModel`](@ref). This enables using general nonlinear programming and trust-region solvers—such as `JSOSolvers.jl` (`trunk`, `tron`), `Ipopt.jl`, and `Optim.jl`—on problems formulated with `NLLSModel`.

---

## Quick Start

```julia
using NLLSModel
using NLPModels
using JSOSolvers
using StaticArrays, Static

# 1. Define residual blocks
struct Rosenbrock1 <: AbstractResidual end
NLLSModel.ndeps(::Rosenbrock1) = static(1)
NLLSModel.nres(::Rosenbrock1) = static(1)
NLLSModel.varindices(::Rosenbrock1) = SVector(1)
NLLSModel.getvars(::Rosenbrock1, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::Rosenbrock1, x) = SVector(1.0 - x[1])
Base.eltype(::Rosenbrock1) = Float64

struct Rosenbrock2 <: AbstractResidual end
NLLSModel.ndeps(::Rosenbrock2) = static(1)
NLLSModel.nres(::Rosenbrock2) = static(1)
NLLSModel.varindices(::Rosenbrock2) = SVector(1)
NLLSModel.getvars(::Rosenbrock2, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::Rosenbrock2, x) = SVector(10.0 * (x[2] - x[1]^2))
Base.eltype(::Rosenbrock2) = Float64

# 2. Build the NLLSModel
prob = NLLSModel(SVector{2, Float64}, Union{Rosenbrock1, Rosenbrock2})
addvariable!(prob, SVector(-1.2, 1.0))
addcost!(prob, Rosenbrock1())
addcost!(prob, Rosenbrock2())

# 3. Wrap as an AbstractNLPModel
nlp = NLPLSModel(prob; name="Rosenbrock")

# 4. Solve using JSOSolvers
stats = trunk(nlp)
println("Status: ", stats.status)
println("Solution: ", stats.solution)

# 5. Synchronize solution back to problem variables
syncvars!(nlp, stats.solution)
println("Optimal variables: ", prob.variables[1])
```

---

## Key Features

### 1. Automatic Sparsity & Decision
`NLPLSModel` inspects the variable structure and problem size:
- **Dense Problems**: Preallocates an $n \times n$ dense Hessian buffer with standard lower-triangular sparse coordinate extraction.
- **Large Block-Sparse Problems**: Uses `NLLSModel` block-sparse matrix structure and derives the exact sparse lower-triangle coordinate pattern, enabling $O(\text{nnz})$ evaluation with zero re-allocations.

### 2. Zero-Allocation Evaluation Caching
Evaluating gradient and Gauss-Newton Hessian simultaneously via $J^T r$ and $J^T J$ is standard in nonlinear least squares. `NLPLSModel` caches the linearization evaluated at $x$:
- When a solver calls `obj(nlp, x)` followed by `grad!(nlp, x, g)` and `hprod!(nlp, x, v, Hv)` at the same point, the cost, gradient, and Hessian are evaluated once and reused.
- Trial step cost evaluations in line searches or trust-region checks evaluate only the residual costs without computing derivatives.

### 3. Hessian-Vector Products
`hprod!` computes $H v = (J^T J) v$ directly using in-place symmetric matrix-vector multiplication (`mul!(Hv, Symmetric(H, :L), v)`), allowing Krylov/CG-based solvers such as `JSOSolvers.trunk` to avoid explicit matrix factorizations.

### 4. Updating Variables with `syncvars!`
Because `NLLSModel` variables can be stored as static vectors, numbers, or custom manifold/bounded variable blocks, [`syncvars!`](@ref) takes the flat solver solution vector and repopulates `prob.variables` in-place.
