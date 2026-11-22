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

Abstract base type for robust kernels `ρ(s)` of the squared residual norm `s = ‖r‖²`. A residual's cost is `0.5·ρ(s)`.
The convention is `ρ(s) = −2 log p(r)`, with `p` a density over `r`; a fixed kernel may drop any constant term
(`NoRobust` is `ρ(s) = s`). A kernel defines [`robustify`](@ref), and may define [`robustifydcost`](@ref) (otherwise
computed by automatic differentiation).
"""
AbstractRobustifier

"""
    AbstractAdaptiveRobustifier <: AbstractRobustifier

Abstract base type for robust kernels with parameters that can be optimized: a variable of the model, the first
variable of an [`AbstractAdaptiveResidual`](@ref). An adaptive kernel returns `ρ(s) = −2 log p(r)` exactly, with every
term that depends on its parameters or on the residual length `d`, which it carries (as `static(d)` in its
constructor; [`addcost!`](@ref) checks it against `nres`). A density can exceed 1, so its cost can be negative. See
[`ContaminatedGaussian`](@ref) and [`GaussianPlusUniform`](@ref).
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
    ContaminatedGaussian(s1, s2, w, d::StaticInt)

Adaptive robust kernel for residuals of length `d`: a mixture of two zero-mean Gaussians of standard deviations `s1` and
`s2`, with weights `w` and `1 − w`,

`ρ(s) = −2 log(w N_d(s; s1) + (1 − w) N_d(s; s2))`, where `N_d(s; σ) = (2πσ²)^(−d/2) exp(−s/(2σ²))` and `s = ‖r‖²`.

`s1`, `s2` and `w` are the kernel's variables (optimized as `log(1/s1)`, `log(1/s2)` and `logit(w)`); the arguments are
promoted to one float type. The components are kept in the order given: the density is the same with `(s1, w)` and
`(s2, 1 − w)` exchanged, so which is the narrow one is the caller's choice. Pass `d` as `static(d)`: it must equal `nres`
of the residuals that use the kernel, which [`addcost!`](@ref) checks. As `ρ` is `−2 log p` exactly, a cost can be
negative.

# Example
```julia
using Static
kernel = ContaminatedGaussian(1.0, 10.0, 0.8, static(1))
robustify(kernel, 0.5)
```
"""
ContaminatedGaussian

"""
    GaussianPlusUniform(sigma, w, d::StaticInt; volume)

Adaptive robust kernel for residuals of length `d`: a Gaussian of standard deviation `sigma` and weight `w`, plus a
uniform outlier density `(1 - w) / A` with `A = volume`,

`ρ(s) = −2 log(w N_d(s; sigma) + (1 − w)/A)`, where `N_d(s; σ) = (2πσ²)^(−d/2) exp(−s/(2σ²))` and `s = ‖r‖²`.

`sigma` and `w` are the kernel's variables (optimized as `log(1/sigma)` and `logit(w)`). The outlier volume `A` is
supplied, finite and positive, and never optimized; a residual can hand over its own with [`withlogvolume`](@ref). The
uniform term is not truncated to a region: the kernel sees only `s`. Pass `d` as `static(d)`: it must equal `nres` of the
residuals that use the kernel, which [`addcost!`](@ref) checks. As `ρ` is `−2 log p` exactly, a cost can be negative.

# Example
```julia
using Static
kernel = GaussianPlusUniform(1.0, 0.9, static(2); volume = 100.0)
robustify(kernel, 0.5)
```
"""
GaussianPlusUniform

"""
    withlogvolume(kernel::GaussianPlusUniform, logvolume) -> GaussianPlusUniform

The same kernel with the logarithm of its outlier volume replaced by `logvolume` (converted to the kernel's element
type): a copy of the fields, of the same concrete type. A residual whose outliers have a volume of their own hands it
over this way, from `getvars` (an adaptive kernel) or `robustkernel` (a fixed one).

# Example
```julia
using Static
kernel = GaussianPlusUniform(1.0, 0.9, static(2); volume = 100.0)
withlogvolume(kernel, log(250.0)) == GaussianPlusUniform(1.0, 0.9, static(2); volume = 250.0) # true
```
"""
withlogvolume

"""
    optimizerobustifier(kernel, s[, logvolume]; maxiters = 10, rtol = 1e-6, wprior = (1.0, 1.0)) -> kernel

Fit an adaptive kernel ([`ContaminatedGaussian`](@ref) or [`GaussianPlusUniform`](@ref)) to the squared residual norms
`s` (`sᵢ = ‖rᵢ‖²`) by expectation-maximization, and return the fitted kernel, of the same type; `kernel` is the start.
For a `GaussianPlusUniform`, `logvolume` (of the length of `s`) gives each residual's outlier log volume `log Aᵢ`;
omitted, every residual uses the kernel's own, which the fitted kernel keeps.

- `maxiters`: the most iterations.
- `rtol`: stop when the kernel's parameters change by less than this, relatively.
- `wprior = (α, β)`: a Beta prior on the first component's weight `w`, as pseudo-counts; both finite and at least 1.
  `(1, 1)` is no prior.

It also stops when the responsibilities are all in one component or a scale would not be finite and positive (all the
inlier residuals zero, say), returning the kernel it has; an empty `s` returns `kernel`. It allocates nothing.

# Example
```julia
using Static
s = [0.1, 0.5, 2.0, 0.3, 80.0, 0.7, 1.2, 150.0]
kernel = optimizerobustifier(ContaminatedGaussian(0.5, 5.0, 0.5, static(1)), s)
```
"""
function optimizerobustifier end

"""
    optimizerobustifier!(model::Model, index; s = Float64[], logvolume = Float64[], maxiters = 10, rtol = 1e-6,
                         wprior = (1.0, 1.0)) -> cost

Fit the adaptive kernel `model.variables[index]` (a [`ContaminatedGaussian`](@ref) or a [`GaussianPlusUniform`](@ref))
to the residuals that use it, at the model's current variables, with [`optimizerobustifier`](@ref); write it back, and
return the model's new [`cost`](@ref). The residuals are the [`AbstractAdaptiveResidual`](@ref)s whose first variable
is `index`; for a `GaussianPlusUniform` each residual's outlier log volume is that of the kernel its [`getvars`](@ref)
hands over (see [`withlogvolume`](@ref)). A residual holding a fixed kernel through [`robustkernel`](@ref) is not
visited. With no residual using the kernel, it is unchanged.

`s` and `logvolume` are buffers, resized to the number of residuals: a caller who keeps them allocates nothing after
the first call. The other keywords are those of `optimizerobustifier`.

To alternate between fitting the kernel and optimizing the other variables, fix the kernel and fit it in the
optimizer's callback, with `OptimizationOptions(callbackchangesstate = true)`:
`callback = (cost, x, data, iteratedata) -> (optimizerobustifier!(model, index; s, logvolume), 0)`.
"""
function optimizerobustifier! end

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
    optimvars(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, min_work=$MIN_WORK_PER_THREAD_HESSIAN, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64) -> (f, fdf!, x0, H)

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
    robustifydkernel(kernel::AbstractAdaptiveRobustifier, squared_norm) -> (rho, gradient, hessian)

Computes `ρ(s)` and its derivatives with respect to the kernel's parameters followed by `s = ‖r‖²`: the gradient, an
`SVector` of length `nvars(kernel) + 1`, and the Hessian, a symmetric `SMatrix` of that size, in the same order. The
parameters are those `update(kernel, x)` steps (for the kernels here, logarithms and logits). Computed by automatic
differentiation unless the kernel defines it.
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
Where the Jacobian is static, the Hessian returned is not an `Array` or `SArray` but a lazy `AbstractMatrix` that
computes each entry when it is read; index it, or `collect` it, for a stored copy. A method of your own may return any
`AbstractVector` gradient and `AbstractMatrix` Hessian, ordinary arrays included.
"""
function computecostgradhess end

"""
    NLPLSModel(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, min_work=$MIN_WORK_PER_THREAD_HESSIAN, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64, name="NLPLSModel")

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
    NLLSNLSModel(problem::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), reduction=:auto, storage=:auto, max_dense_bytes=$MAX_DENSE_BYTES, min_work=$MIN_WORK_PER_THREAD_JACOBIAN, min_work_cost=$MIN_WORK_PER_THREAD_COST, precision=Float64, name="NLLSNLSModel")

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

It is built on the Jacobian path ([`optimvars_nls`](@ref)): `reduction = :auto` means `:none`. The cost alone (`obj`) is
evaluated by the cost function, threaded by `min_work_cost`; the residual and Jacobian by the Jacobian fill, threaded by
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

Recommends how to solve `model`, from its structure and its current values: a formulation (`:hessian`, the Hessian, or `:jacobian`, the residual and the Jacobian), the `reduction` and `storage`, a linear solver and an iterator,
each chosen by a hand-written rule. Displaying the result explains each choice and prints the equivalent call to
`optimize!(model, options; ...)`; with SecondOrderOptimization loaded, `optimize!(model, rec)` runs it.

It builds the Hessian path as `optimize!(model, options)` does with `reduction = storage = :auto`, then builds, linearizes
and factorizes its linear system once, at the model's current values, to measure the spread of the Hessian's diagonal
(largest over smallest), a lower bound on the condition number of the Jacobi-scaled Hessian `D^-½ H D^-½` (by
$REC_INVERSE_ITERATIONS steps of inverse iteration), and the time of a cost evaluation, a linearization and an undamped
solve. When every cost is an [`AbstractResidual`](@ref) and the Hessian is ill-conditioned (the estimate is at least
$(REC_JACOBIAN_CONDITION)) or could not be factorized undamped, it also tests the rank of the Jacobian `J`: it takes the
weakest direction `w` of the scaled Hessian (from the undamped factorization, or else from the first of the scaled
dampings `H + λD`, λ in $(REC_RANK_DAMPINGS), that factorizes), refines it by $(REC_RANK_REFINEMENTS) step of inverse
iteration on `JᵀJ`, with products of `J`, and reads `t = ‖J D^-½ w‖`, with `‖w‖ = 1`: the smallest of the readings
before and after each step.
The column-scaled `J D^-½` has unit columns, so `t` is an upper bound on its smallest singular value. This evaluates
the Jacobian once, through [`optimvars_nls`](@ref) without elimination; the first such call compiles the Jacobian
evaluators. Then:

- **Formulation:** `:hessian` when the Hessian is well conditioned, or when a cost is not a residual, except for the
  normal equations from the Jacobian, below. Otherwise the rank
  test decides: `J` is numerically rank-deficient when `t` is below $(REC_RANK_TOLERANCE) (a gauge freedom, for example),
  which factorizing `J` would not help and damping handles, so `:hessian`; otherwise `J` has full numerical rank, and
  `:jacobian`, since forming `JᵀJ` squares its conditioning. Where the test cannot run, `:hessian`: the Hessian has a
  diagonal entry that is not positive, no damping tried factorizes it (it is indefinite, say), or [`optimvars_nls`](@ref)
  cannot build the model (an unfixed adaptive kernel, for example).
  *The normal equations from the Jacobian:* `:jacobian` with the normal equations formed from a dense Jacobian when every
  cost is a residual, the Hessian is well conditioned (no rank test), the Hessian path's system is dense (not static,
  sparse or eliminated), the dimension is above $(MAX_STATIC_SYSTEM), and building the Hessian cost by cost is estimated
  to take at least $(REC_NORMAL_MARGIN) times the multiply-adds of one product `JᵀJ`: `Σ n_c² (m_c + $(REC_HESSIAN_SCATTER))`
  over the costs (`m_c` residual rows, `n_c` unfixed dimensions; $(REC_HESSIAN_SCATTER) stands for one load-add-store into
  the Hessian) against `M N² / 2` for the `M × N` Jacobian; and the dense Jacobian fits `max_dense_bytes` and
  [`optimvars_nls`](@ref) builds it.
- **Reduction and storage:** what `:auto` builds, on the formulation's path; for the normal equations from the Jacobian,
  `reduction = :none` and `storage = :dense`, which the product needs.
- **Linear solver:** on the Hessian path, `:cholesky` for a static, dense or eliminated system and `:ldl` for a sparse
  one (what `linsolver = :auto` uses), but `:ldl` whenever a cost is not a residual, since the Hessian can then be
  indefinite; on the Jacobian path `:qr`, and `:cholesky` (of `JᵀJ`, formed from the Jacobian in one product) for the
  normal equations from the Jacobian.
- **Iterator:** `:dogleg` when the undamped solve succeeded, the condition estimate is below $(REC_DOGLEG_CONDITION), the
  diagonal's spread is below $(REC_DIAGSPREAD) and a solve takes at least as long as a linearization; otherwise
  `:levenbergmarquardt_dtd` when the spread is at least $(REC_DIAGSPREAD); otherwise `:levenbergmarquardt`.

Every threshold is a starting value, not yet measured against benchmarks ($(REC_HESSIAN_SCATTER) and $(REC_NORMAL_MARGIN)
too). The model's variables are not changed. A model whose cost, gradient or Hessian is not finite at its current values
is an `ArgumentError`.

The two work estimates count arithmetic only: not threads (the Hessian is filled cost by cost on Polyester's threads, the
product `JᵀJ` by BLAS), and not a cost whose Jacobian block is mostly zeros. The iterator's test of a solve against a
linearization still reads the Hessian path's times.

The rank test detects a null space of `J`; it is not an estimate of `J`'s condition number. Its known limits: a null
space together with a remainder whose smallest scaled singular value is below about 1e-7 reads as full rank, and goes to
the Jacobian path; and a full-rank `J` with a condition number near 1e13 reads about 1e-8 when damping was needed, so it
too goes to the Jacobian path.

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
