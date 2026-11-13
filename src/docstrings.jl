"""
    Model{VarTypes, CostTypes}([variables])
    Model(VT, CT)
    Model(variables, CT)

Bipartite graph nonlinear least squares problem representation holding costs and variables.

# Fields
- `costs`: Container of cost and residual blocks that define the problem.
- `variables::Vector{VarTypes}`: Vector of variable blocks.
- `varcostmap::SparseMatrixCSC{Bool, Int}`: Variable-to-cost bipartite incidence matrix.
- `varcostmapvalid::Bool`: Flag tracking validity of `varcostmap`.

# Example
```julia
problem = Model(Float64, MyResidual)
addvariable!(problem, 0.0)
addcost!(problem, MyResidual(1.0))
```
"""
Model

"""
    AbstractCost

Abstract supertype for all cost blocks in a [`Model`](@ref).
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
    addvariable!(problem::Model, variable) -> Int

Appends a variable block to `problem` and returns its 1-based index.

# Example
```julia
problem = Model(Float64, MyResidual)
idx = addvariable!(problem, 1.0) # 1
```
"""
addvariable!

"""
    addcost!(problem::Model, cost::AbstractCost)
    addresidual!(problem::Model, res::AbstractResidual)

Appends a cost or residual block to `problem`.

# Example
```julia
addcost!(problem, my_residual)
```
"""
addcost!

"""
    optimvars(problem::Model, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()); reduction=:auto, reduced=:auto, min_work=$MIN_WORK_FOR_THREADED_LINEARIZATION, min_work_cost=$MIN_WORK_FOR_THREADED_COST) -> (f, fdf!, x0, H)

The normal way to solve a `Model` is `optimize!(model, options)` (needs `using SecondOrderOptimization`); `optimvars` is the
advanced route, for custom outer loops, reusing a workspace, or other solvers.

Constructs the objective function `f`, in-place gradient/Hessian evaluator `fdf!`, initial variable state `x0`, and Hessian template `H` (`nothing` for static/dense systems or a `SparseMatrixCSC` for sparse systems) for `problem`.

Wrap the returned 4-tuple in a `HessianObjective` and pass it to `SecondOrderOptimization.optimize!`, then write the
solution back into the model:
```julia
f, fdf!, x0, H = optimvars(model)
result = optimize!(x0, HessianObjective(f, fdf!; H), options)
syncvars!(model, fdf!, x0)
```
(`optimize!` needs a mutable `x0`; for a single variable of static size use `optimize`, which returns `(x, result)`, and
sync `x`.)

# Arguments
- `problem::Model`: The nonlinear least squares problem.
- `unfixed`: Optional variable index (`Integer`), variable type (`DataType`), or boolean mask specifying which variables to optimize (defaults to `nothing`, optimizing all variables).
- `numthreads`: Number of threads to use for parallel cost and derivative evaluation (`StaticInt` or `Integer`).
- `reduction`: whether eliminable variables are eliminated (Schur complement). `:auto` (default) eliminates when the model has `iseliminable` variables and the structure fits, and falls back to no elimination otherwise; `:eliminate` eliminates or throws an `ArgumentError`; `:none` never eliminates. Any other value is an `ArgumentError`. The returned `H` is a `SchurBlockMatrix` exactly when elimination happens, and the optimizer eliminates exactly when it is handed one.
- `reduced`: storage of the reduced system when elimination happens: `:auto` (default; chosen from the predicted fill of a Cholesky factor), `:dense` or `:sparse`. It is silently ignored (no error, no warning) when no elimination happens, that is with `reduction = :none`, or with `:auto` on a model that does not eliminate.
- `min_work`: Linearization is threaded only if the problem's estimated work (see `costwork`: residual length times variables squared, summed over costs; one scalar residual of one variable is one unit) is at least this.
- `min_work_cost`: The same floor for evaluating the cost alone, which is cheaper per unit of work and so needs more of it to pay for threading.
"""
optimvars

"""
    cost(problem::Model, numthreads=StaticInt(Threads.nthreads())) -> Float64

Evaluates and returns the total scalar objective cost of `problem`.

# Example
```julia
c = cost(problem)
```
"""
cost

"""
    subproblem(problem::Model, unfixed) -> Model

Extracts a subproblem containing only costs connected to `unfixed` variables.
"""
subproblem

"""
    subproblem!(to::Model, from::Model, costindices) -> to

Populates subproblem `to` in-place with the subset of costs from `from` indexed by `costindices`.
"""
subproblem!

# Extensible user-specialized methods
"""
    ndeps(cost::AbstractCost) -> StaticInt

Number of variable blocks `cost` depends upon (returned as `static(N)`).

# Example
```julia
NLLSModel.ndeps(::MyResidual) = static(2)
```
"""
function ndeps end

"""
    nres(cost::AbstractResidual) -> StaticInt or Int

Dimensionality of the residual vector returned by `cost` (e.g. `static(M)` or an `Int`).

Returning a `StaticInt` makes the two static kernels available (see "Performance" in the README):
the full static multiply, which also needs every unfixed variable to have a static size
(total `n <= MAX_STATIC_VAR`) and `M * n <= MAX_STATIC_JAC`; and, for costs that are not on that path, the
static-column kernel, which needs `M <= MAX_STATIC_ROWS`. A run-time length (an `Int`) always uses the `@turbo`
kernel (or plain loops for a Jacobian that is not a strided array of hardware floats).

# Example
```julia
NLLSModel.nres(::MyResidual) = static(3)
```
"""
function nres end

"""
    varindices(cost::AbstractCost) -> SVector or Int

Indices of variables in `problem.variables` that `cost` depends upon.

# Example
```julia
NLLSModel.varindices(r::MyResidual) = r.varind
```
"""
function varindices end

"""
    getvars(cost::AbstractCost, vars::Vector) -> Tuple

Extracts the concretely-typed tuple of variables for `cost` from `vars`.

# Example
```julia
NLLSModel.getvars(r::MyResidual, vars::Vector) = (vars[r.varind[1]]::Float64, vars[r.varind[2]]::Float64)
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
NLLSModel.robustkernel(r::MyRobustResidual) = Huber2oKernel(1.5)
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

Residuals are evaluated on several threads at once, so this function (like [`computeresjac`](@ref) and
[`computecostgradhess`](@ref)) must itself be single-threaded: no BLAS on large matrices and no `Threads.@threads`.
If it needs BLAS, call `BLAS.set_num_threads(1)` in the application.

# Example
```julia
NLLSModel.computeresidual(r::MyResidual, x, y) = y - x - r.offset
```
"""
function computeresidual end

"""
    computeresjac(varflags, cost::AbstractResidual, vars...) -> (res, jac)

Computes the residual vector and Jacobian matrix for active variables indicated by bitmask `varflags`.
Defaults to ForwardDiff automatic differentiation when not specialized.

Rules for user code, because residuals are evaluated on several threads at once:
- Arrays returned from `computeresjac` (or [`computecostgradhess`](@ref)) must not be shared with other residuals.
  Returning arrays that the residual owns or that are local to the call is fine; a buffer reused by two residuals
  evaluated at the same time is not.
- The code must itself be single-threaded (no BLAS on large matrices, no `Threads.@threads`). If it needs BLAS, call
  `BLAS.set_num_threads(1)` in the application.

The residual can be a `Number`, a `Vector` or any `StaticVector` (`SVector`, `MVector`, ...). The Jacobian can be an
`SMatrix`, a `Matrix` (or any strided matrix of hardware floats) or, for a scalar residual, the adjoint of the gradient
vector. The two need not both be static or both dynamic: a static residual with a `Matrix` Jacobian is fine, as is a
`Vector` residual with an `SMatrix`. The kernels for products of the Jacobian are chosen by the types of the residual and
Jacobian and by the type of `nres` (see [`nres`](@ref)).
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

The same rules apply as for [`computeresjac`](@ref): returned arrays must not be shared with other residuals, and the
code must be single-threaded. For an [`AbstractResidual`](@ref) the gradient and Hessian are computed from the Jacobian
without BLAS; the arrays returned alias per-task buffers and are valid until the same task evaluates its next cost.
"""
function computecostgradhess end

"""
    NLPLSModel(problem::Model; unfixed=nothing, numthreads=1, name="NLPLSModel")

Wraps a [`Model`](@ref) as an `NLPModels.AbstractNLPModel` for use with general
nonlinear programming solvers such as `JSOSolvers`, `Ipopt`, and `Optim`.

Requires loading `NLPModels`:
```julia
using NLLSModel, NLPModels
nlp = NLPLSModel(problem)
```

# Arguments
- `problem::Model`: The nonlinear least-squares problem to wrap.

# Keyword Arguments
- `unfixed = nothing`: Boolean mask, integer, or vector indicating which variables are unfixed (optimized).
- `numthreads = 1`: Number of threads for cost and gradient/Hessian evaluation.
- `name = "NLPLSModel"`: Name string assigned to the model metadata.

# Features
- Automatically selects dense or block-sparse Hessian representations based on problem structure.
- Implements `obj`, `grad!`, `objgrad!`, `hess_structure!`, `hess_coord!`, and `hprod!`.
- Caches cost, gradient, and Gauss-Newton Hessian evaluations across queries at the same point.
- Transfers solution vectors back to `problem.variables` via [`syncvars!`](@ref).
"""
NLPLSModel

"""
    syncvars!(nlp, x::AbstractVector) -> Model
    syncvars!(prob::Model, nlp, x::AbstractVector) -> Model

Copies the flat solution vector `x` back into the problem's `variables` array,
taking unfixed variable subsets and block variable structures into account.
"""
syncvars!
