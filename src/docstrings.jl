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

Appends a cost or residual block to `problem`.

# Example
```julia
addcost!(problem, my_residual)
```
"""
addcost!

"""
    optimvars(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, min_work=$MIN_WORK_PER_THREAD_NORMAL, min_work_cost=$MIN_WORK_PER_THREAD_COST) -> (f, fdf!, x0, H)

The normal way to solve a `Model` is `optimize!(model, options)` (needs `using SecondOrderOptimization`); `optimvars` is the
advanced route, for custom outer loops, reusing a workspace, or other solvers.

Constructs the objective function `f`, in-place gradient/Hessian evaluator `fdf!`, initial variable state `x0`, and Hessian template `H` for `problem`.

`H` is a prototype of the Hessian `fdf!(g, H, x)` fills, of the type the linear system stores it in:
- one unfixed variable: an `MMatrix{N, N, Float64}` when its size `N` is static and at most 16, otherwise a
  `Matrix{Float64}`, zero-filled. The gradient buffer `g` is the matching vector type, an `MVector{N, Float64}` or a
  `Vector{Float64}`, which `zero(H[:, 1])` builds. Buffers of exactly these types (`deepcopy(H)` and `zero(H[:, 1])`)
  are filled in place; any other `AbstractMatrix`/`AbstractVector` is filled through an internal copy on every call.
- several variables: a `SchurBlockHessian` when variables are eliminated, a `SparseMatrixCSC{Float64, Int}` for a sparse
  system, and `nothing` for a dense one (pass a `Matrix{Float64}` and a `Vector{Float64}`).

Wrap the returned 4-tuple in a `HessianObjective` and pass it to `SecondOrderOptimization.optimize!`, then write the
solution back into the model:
```julia
f, fdf!, x0, H = optimvars(model)
result = optimize!(x0, HessianObjective(f, fdf!; H), options)
syncvars!(model, fdf!, x0)
```
(`optimize!` needs a mutable `x0`; for a single variable of static size use `optimize`, which returns `(x, result)`, and
sync `x`.)

`reduction` and `storage` together decide `H`: it is a `SchurBlockHessian` exactly when elimination happens, and the
optimizer eliminates exactly when it is handed one. Under elimination, `storage = :auto` chooses the reduced system's
storage from the predicted fill of a Cholesky factor.

$KEYWORDS_DOC"""
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

Extracts a subproblem containing only the costs that touch an unfixed variable. `unfixed` takes the forms that the
`unfixed` keyword of `optimize!` and [`optimvars`](@ref) takes: a variable index, a variable type, a Boolean mask over the
variables, a list of variable indices, or `nothing` (all variables). The subproblem shares `problem`'s variables.
"""
subproblem

"""
    subproblem!(to::Model, from::Model, costindices) -> to

Populates subproblem `to` in-place with the subset of costs from `from` indexed by `costindices`: indices into all the
costs of `from`, counted across its cost types in order. They must be sorted in ascending order and between 1 and the
number of costs; otherwise an `ArgumentError` is thrown.
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
static-column kernel, which needs `M <= MAX_STATIC_ROWS`. A run-time length (an `Int`) uses the `@turbo` kernel, or
plain loops for a single-column Jacobian and for a Jacobian that is not a strided array of hardware floats.

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
    NLPLSModel(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, min_work=$MIN_WORK_PER_THREAD_NORMAL, min_work_cost=$MIN_WORK_PER_THREAD_COST, name="NLPLSModel")

Wraps a [`Model`](@ref) as an `NLPModels.AbstractNLPModel` for use with general
nonlinear programming solvers such as `JSOSolvers`, `Ipopt`, and `Optim`.

Requires loading `NLPModels`:
```julia
using NLLSModel, NLPModels
nlp = NLPLSModel(problem)
```

# Arguments
- `problem::Model`: The nonlinear least-squares problem to wrap.

Every unfixed variable must update by plain addition and pack into a flat `Float64` vector (as for
[`flat_optimvars_nls`](@ref)), since the solver steps by adding to the flat vector of the variables' values; otherwise it
throws an `ArgumentError`, before anything is evaluated.

It works on the full system: `reduction = :auto` means `:none`, and `:eliminate` is an `ArgumentError`. `name` is the name
given to the model's metadata (default `"NLPLSModel"`).

# Features
- Automatically selects dense or block-sparse Hessian representations based on problem structure.
- Implements `obj`, `grad!`, `objgrad!`, `hess_structure!`, `hess_coord!`, and `hprod!`.
- Caches cost, gradient, and Gauss-Newton Hessian evaluations across queries at the same point.
- Transfers solution vectors back to `problem.variables` via [`syncvars!`](@ref).

$KEYWORDS_DOC"""
NLPLSModel

"""
    NLLSNLSModel(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work=$MIN_WORK_PER_THREAD_TALL, min_work_cost=$MIN_WORK_PER_THREAD_COST, name="NLLSNLSModel")

Wraps a [`Model`](@ref) as an `NLPModels.AbstractNLSModel{Float64, Vector{Float64}}`, which exposes the residual vector and
its Jacobian, for nonlinear least-squares solvers that work on them directly, such as `JSOSolvers.trunk` and
`JSOSolvers.tron`.

Requires loading `NLPModels`:
```julia
using NLLSModel, NLPModels
nls = NLLSNLSModel(problem)
```

Every cost must be a non-adaptive [`AbstractResidual`](@ref) with a linear kernel (`NoRobust`, or `Scaled` with a finite non-negative height), so that the
cost is exactly `½‖r‖²`, which these solvers derive from `r`; otherwise it throws an `ArgumentError`. Every unfixed
variable must update by plain addition and pack into a flat `Float64` vector (as for [`flat_optimvars_nls`](@ref)).

It is built on the tall path ([`optimvars_nls`](@ref)): `reduction = :auto` means `:none`. The cost alone (`obj`) is
evaluated by the cost function, threaded by `min_work_cost`; the residual and Jacobian by the tall fill, threaded by
`min_work`. `name` is the name given to the model's metadata (default `"NLLSNLSModel"`).

The solution is transferred back to `problem.variables` with [`syncvars!`](@ref).

$KEYWORDS_DOC"""
NLLSNLSModel

"""
    syncvars!(nlp, x::AbstractVector) -> Model
    syncvars!(prob::Model, nlp, x::AbstractVector) -> Model

Copies the flat solution vector `x` back into the problem's `variables` array,
taking unfixed variable subsets and block variable structures into account.
"""
syncvars!

"""
    recommendoptimizer(model::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), max_dense_bytes=$MAX_DENSE_BYTES) -> OptimizationRecommendation

Recommends how to solve `model`, from its structure and its current values: a formulation (`:hessian`, the normal
equations, or `:jacobian`, the residual and the Jacobian), the `reduction` and `storage`, a linear solver and an iterator,
each chosen by a hand-written rule. Displaying the result explains each choice and prints the equivalent call to
`optimize!(model, options; ...)`; with SecondOrderOptimization loaded, `optimize!(model, rec)` runs it.

It builds the normal path as `optimize!(model, options)` does with `reduction = storage = :auto`, then builds, linearizes
and factorizes its linear system once, at the model's current values, to measure the spread of the Hessian's diagonal
(largest over smallest), a lower bound on the condition number of the Jacobi-scaled Hessian `D^-½ H D^-½` (by
$REC_INVERSE_ITERATIONS steps of inverse iteration), and the time of a cost evaluation, a linearization and an undamped
solve. Then:

- **Formulation:** `:jacobian` when every cost is an [`AbstractResidual`](@ref) and the condition estimate is finite and at
  least $(REC_TALL_CONDITION); otherwise `:hessian`. A system that does not factorize (rank-deficient or indefinite) has
  no estimate and stays `:hessian`. If [`optimvars_nls`](@ref) cannot build the model (an unfixed adaptive kernel, for
  example), `:hessian`.
- **Reduction and storage:** what `:auto` builds, on the formulation's path.
- **Linear solver:** on the normal path, `:cholesky` for a static, dense or eliminated system and `:ldl` for a sparse
  one (what `linsolver = :auto` uses), but `:ldl` whenever a cost is not a residual, since the Hessian can then be
  indefinite; on the tall path `:qr`.
- **Iterator:** `:dogleg` when the undamped solve succeeded, the condition estimate is below $(REC_DOGLEG_CONDITION), the
  diagonal's spread is below $(REC_DIAGSPREAD) and a solve takes at least as long as a linearization; otherwise
  `:levenbergmarquardt_dtd` when the spread is at least $(REC_DIAGSPREAD); otherwise `:levenbergmarquardt`.

Every threshold is a starting value, not yet measured against benchmarks. The model's variables are not changed. A model
whose cost, gradient or Hessian is not finite at its current values is an `ArgumentError`.

The keywords are those of [`optimvars`](@ref), with the same meanings and defaults; the recommendation keeps them, and
`optimize!(model, rec)` uses them.

# Example
```julia
rec = recommendoptimizer(model)   # displays the explanation and the call
result = optimize!(model, rec)    # with `using SecondOrderOptimization`
```
"""
recommendoptimizer

"""
    OptimizationRecommendation

What [`recommendoptimizer`](@ref) recommends for a model: a formulation, reduction, storage, linear solver and iterator,
the keywords it was made with, and the evidence for each choice (the features measured and the rule that fired).
Displaying it in the REPL (`show` with `MIME"text/plain"`) explains each choice and prints the equivalent call to
`optimize!(model, options; ...)`; the two-argument `show` is one line with the five choices. With SecondOrderOptimization
loaded, `optimize!(model, rec)` and `optimize(model, rec)` run it.

A recommendation holds for the model, and the values, it was made from; it is not checked against the model when it is
used. Its fields are internal, and it cannot be changed: to change a choice, set the options and keywords by hand on
`optimize!(model, options; ...)`, starting from the call it prints.
"""
OptimizationRecommendation
