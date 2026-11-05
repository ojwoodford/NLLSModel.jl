"""
    NLLSProblem{VarTypes, CostTypes}([variables])
    NLLSProblem(VT, CT)
    NLLSProblem(variables, CT)

Bipartite graph nonlinear least squares problem representation holding costs and variables.

# Fields
- `costs`: Container of cost and residual blocks that define the problem.
- `variables::Vector{VarTypes}`: Vector of variable blocks.
- `varcostmap::SparseMatrixCSC{Bool, Int}`: Variable-to-cost bipartite incidence matrix.
- `varcostmapvalid::Bool`: Flag tracking validity of `varcostmap`.

# Example
```julia
problem = NLLSProblem(Float64, MyResidual)
addvariable!(problem, 0.0)
addcost!(problem, MyResidual(1.0))
```
"""
NLLSProblem

"""
    AbstractCost

Abstract supertype for all cost blocks in an [`NLLSProblem`](@ref).
"""
AbstractCost

"""
    AbstractResidual <: AbstractCost

Abstract supertype for least-squares residual blocks `r(x)`, contributing `0.5 * robustify(kernel, ||r(x)||^2)` to the cost.
"""
AbstractResidual

"""
    AbstractAdaptiveResidual <: AbstractResidual

Residual with learnable robustifier kernel parameters as first variable.
"""
AbstractAdaptiveResidual

"""
    AbstractRobustifier

Abstract base type for M-estimator loss functions `rho(s)` where `s = ||r||^2`.
"""
AbstractRobustifier

"""
    AbstractAdaptiveRobustifier <: AbstractRobustifier

Abstract base type for robustifiers whose shape parameters are optimized during solve.
"""
AbstractAdaptiveRobustifier

"""
    NoRobust()

Identity loss function `rho(s) = s` (standard squared loss).

# Example
```julia
kernel = NoRobust()
robustify(kernel, 4.0) # 4.0
```
"""
NoRobust

"""
    Scaled(kernel::AbstractRobustifier, height::Real)

Scales another robustifier `kernel` by `height`.

# Example
```julia
kernel = Scaled(HuberKernel(1.0), 2.0)
```
"""
Scaled

"""
    HuberKernel(width)

First-order Huber M-estimator with linear tails beyond `width`.

# Example
```julia
kernel = HuberKernel(1.5)
robustify(kernel, 4.0)
```
"""
HuberKernel

"""
    Huber2oKernel(width)

Second-order Huber M-estimator (includes second derivative of `rho(s)` in the Hessian approximation).

# Example
```julia
kernel = Huber2oKernel(1.5)
```
"""
Huber2oKernel

"""
    GemanMcclureKernel(width)

First-order Geman-McClure robust loss kernel `s * w^2 / (s + w^2)`.

# Example
```julia
kernel = GemanMcclureKernel(2.0)
```
"""
GemanMcclureKernel

"""
    GemanMcclure2oKernel(width)

Second-order Geman-McClure robust loss kernel (includes second derivative of `rho(s)` in the Hessian approximation).

# Example
```julia
kernel = GemanMcclure2oKernel(2.0)
```
"""
GemanMcclure2oKernel

"""
    ContaminatedGaussian(s1, s2, w)

Adaptive two-component Gaussian mixture robustifier with standard deviations `s1`, `s2` (`s2 > s1 > 0`) and inlier weight `w` in `(0, 1)`.

# Example
```julia
kernel = ContaminatedGaussian(1.0, 10.0, 0.8)
```
"""
ContaminatedGaussian

"""
    SimpleError2{V1, V2}(measurement, var1, var2)

Two-variable residual computing `generatemeasurement(v1, v2) - measurement`.

# Example
```julia
err = SimpleError2{SVector{3, Float64}, SVector{3, Float64}}(SVector(1.0, 2.0), 1, 2)
```
"""
SimpleError2

"""
    SimpleError3{V1, V2, V3}(measurement, var1, var2, var3)

Three-variable residual computing `generatemeasurement(v1, v2, v3) - measurement`.
"""
SimpleError3

"""
    SimpleError4{V1, V2, V3, V4}(measurement, var1, var2, var3, var4)

Four-variable residual computing `generatemeasurement(v1, v2, v3, v4) - measurement`.
"""
SimpleError4

"""
    addvariable!(problem::NLLSProblem, variable) -> Int

Appends a variable block to `problem` and returns its 1-based index.

# Example
```julia
problem = NLLSProblem(Float64, MyResidual)
idx = addvariable!(problem, 1.0) # 1
```
"""
addvariable!

"""
    addcost!(problem::NLLSProblem, cost::AbstractCost)
    addresidual!(problem::NLLSProblem, res::AbstractResidual)

Appends a cost or residual block to `problem`.

# Example
```julia
addcost!(problem, my_residual)
```
"""
addcost!

"""
    optimvars(problem::NLLSProblem, unfixed=nothing, numthreads=StaticInt(Threads.nthreads())) -> (f, fdf!, x0, H)

Constructs the objective function `f`, in-place gradient/Hessian evaluator `fdf!`, initial variable state `x0`, and Hessian template `H` (`nothing` for static/dense systems or a `SparseMatrixCSC` for sparse systems) for `problem`.

The returned 4-tuple can be passed directly to `SecondOrderOptimization.optimize`:
```julia
f, fdf!, x0, H = optimvars(problem)
x_opt, result = optimize(f, fdf!, x0, options; H=H)
```

# Arguments
- `problem::NLLSProblem`: The nonlinear least squares problem.
- `unfixed`: Optional variable index (`Integer`), variable type (`DataType`), or boolean mask specifying which variables to optimize (defaults to `nothing`, optimizing all variables).
- `numthreads`: Number of threads to use for parallel cost and derivative evaluation (`StaticInt` or `Integer`).
"""
optimvars

"""
    cost(problem::NLLSProblem, numthreads=StaticInt(Threads.nthreads())) -> Float64

Evaluates and returns the total scalar objective cost of `problem`.

# Example
```julia
c = cost(problem)
```
"""
cost

"""
    subproblem(problem::NLLSProblem, unfixed) -> NLLSProblem

Extracts a subproblem containing only costs connected to `unfixed` variables.
"""
subproblem

"""
    subproblem!(to::NLLSProblem, from::NLLSProblem, costindices) -> to

Populates subproblem `to` in-place with the subset of costs from `from` indexed by `costindices`.
"""
subproblem!

# Extensible user-specialized methods
"""
    ndeps(cost::AbstractCost) -> StaticInt

Number of variable blocks `cost` depends upon (returned as `static(N)`).

# Example
```julia
NLLSModels.ndeps(::MyResidual) = static(2)
```
"""
function ndeps end

"""
    nres(cost::AbstractResidual) -> StaticInt or Int

Dimensionality of the residual vector returned by `cost` (e.g. `static(M)` or an `Int`).

# Example
```julia
NLLSModels.nres(::MyResidual) = static(3)
```
"""
function nres end

"""
    varindices(cost::AbstractCost) -> SVector or Int

Indices of variables in `problem.variables` that `cost` depends upon.

# Example
```julia
NLLSModels.varindices(r::MyResidual) = r.varind
```
"""
function varindices end

"""
    getvars(cost::AbstractCost, vars::Vector) -> Tuple

Extracts the concretely-typed tuple of variables for `cost` from `vars`.

# Example
```julia
NLLSModels.getvars(r::MyResidual, vars::Vector) = (vars[r.varind[1]]::Float64, vars[r.varind[2]]::Float64)
```
"""
function getvars end

"""
    generatemeasurement(vars...)

Generates expected measurement for [`SimpleError2`](@ref), [`SimpleError3`](@ref), and [`SimpleError4`](@ref) residuals.
"""
function generatemeasurement end

"""
    robustkernel(cost::AbstractResidual) -> AbstractRobustifier

Returns the robustifier kernel for `cost` (defaults to [`NoRobust()`](@ref)).

# Example
```julia
NLLSModels.robustkernel(r::MyRobustResidual) = Huber2oKernel(1.5)
```
"""
function robustkernel end

"""
    robustify(kernel::AbstractRobustifier, squared_norm) -> Real

Applies robust loss function `kernel` to squared residual norm `s = ||r||^2`.
"""
function robustify end

"""
    robustifydcost(kernel::AbstractRobustifier, squared_norm) -> (rho, drho_ds, d2rho_ds2)

Computes the robustified squared residual value and its first and second derivatives with respect to `s = ||r||^2`.
"""
function robustifydcost end

"""
    robustifydkernel(kernel::AbstractAdaptiveRobustifier, squared_norm)

Computes derivatives of an adaptive robustifier with respect to both the squared residual norm and the kernel parameters.
"""
function robustifydkernel end

"""
    computeresidual(cost::AbstractResidual, vars...)

Computes the residual vector (or scalar) for `cost` given its unpacked variables `vars...`.

# Example
```julia
NLLSModels.computeresidual(r::MyResidual, x, y) = y - x - r.offset
```
"""
function computeresidual end

"""
    computeresjac(varflags, cost::AbstractResidual, vars...) -> (res, jac)

Computes the residual vector and Jacobian matrix for active variables indicated by bitmask `varflags`.
Defaults to ForwardDiff automatic differentiation when not specialized.
"""
function computeresjac end

"""
    computecost(cost::AbstractCost, vars...) -> Float64

Computes the scalar cost contribution of `cost` given its unpacked variables `vars...`.
"""
function computecost end

"""
    computecostgradhess(varflags, cost::AbstractCost, vars...) -> (cost, grad, hess)

Computes the scalar cost, gradient vector, and Hessian matrix for active variables indicated by bitmask `varflags`.
"""
function computecostgradhess end
