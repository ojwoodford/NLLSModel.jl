# Linear Solvers & Sparsity

In nonlinear least squares, the normal equations at each iteration have the form:

```math
(J^T W J + \lambda D) \Delta \mathbf{x} = -J^T W \mathbf{r}
```

## Automatic Dense vs Sparse Block Selection

`NLLSModel.jl` automatically inspects the variable-cost bipartite graph (`getvarcostmap(problem)`):
- For small or dense problems, it assembles the Hessian into a `BlockDenseMatrix`.
- For larger sparse problems ($\ge 40$ total degrees of freedom with sufficient block sparsity), it assembles the Hessian into a `BlockSparseMatrix` and symmetrifies into a `SparseMatrixCSC`.

## Selecting the Linear Solver Backend

Set the `linsolver` keyword in `OptimizationOptions` (or `NLLSOptions`):

- `:auto` (default): Uses dense Cholesky (with QR fallback) for `BlockDenseMatrix` and static systems, and zero-allocation sparse $LDL^T$ (`LDLFactorizations.jl`) for `BlockSparseMatrix`.
- `:ldl`: Uses `LDLFactorizations.jl` for sparse systems and Bunch-Kaufman symmetric indefinite factorization for dense/static systems.
- `:cholesky`: Uses Cholesky factorization (with QR fallback if not positive definite) for both dense and sparse systems.
- `:qr`: Uses QR factorization for maximum numerical stability on ill-conditioned systems.

```julia
using NLLSModel

options = OptimizationOptions(
    iterator = :levenbergmarquardt_dtd,
    linsolver = :ldl
)
res = optimize!(problem, options)
```

## Eliminating Variables: `reduction` and `storage`

Whether eliminable variables (`iseliminable`, for example bundle-adjustment landmarks) are eliminated by a Schur complement is a
choice about the layout of the linear system, so it is made where the system is built: by keywords of
`optimize!(model, options; reduction, storage)` and of [`optimvars`](@ref), not in `OptimizationOptions`. The optimizer
eliminates exactly when the Hessian prototype it is given is a `SchurBlockHessian`. `linsolver` names only the solver for
the system that is left (the reduced system, after elimination).

- `reduction = :auto` (default) eliminates when the model has eliminable variables and the structure fits (each eliminable
  variable coupled only to retained variables; a variable of any size BlockLinearSystems takes as a block), and falls back to no elimination otherwise.
  `:eliminate` eliminates or throws an `ArgumentError`. `:none` never eliminates. Other values are an `ArgumentError`.
- `storage = :auto` (default) stores the reduced system dense or sparse from the predicted fill of its Cholesky factor;
  `:dense` and `:sparse` force it. With nothing eliminated it chooses the storage of the Hessian itself in the same way:
  `:auto` decides from its block sparsity, `:dense` and `:sparse` force it.
- `max_dense_bytes` (default 256 MiB) caps a dense matrix: `:auto` stores a larger one sparse, and `:dense` throws an
  `ArgumentError` giving its size.

```julia
optimize!(problem, options; reduction = :eliminate, storage = :sparse)
```

The same keywords, with the same rule, apply to `formulation = :tall` (the residual and the Jacobian). There the
eliminated variables are eliminated by QR of their rows of the Jacobian, so `JᵀJ` is not formed for them: the prototype
handed to the optimizer is a `SchurBlockJacobian`, and `storage` chooses the storage of the reduced system the same way.
The tall path also needs each eliminated variable to be in at least one cost.

## Choosing a Configuration Automatically

[`recommendoptimizer`](@ref) chooses the formulation, `reduction`, `storage`, linear solver and iterator for a model by
hand-written rules, from the model's structure and from one linearization and factorization of its normal equations at
the model's current values. That probe measures the spread of the Hessian's diagonal, a lower bound on the condition
number of the Jacobi-scaled Hessian, and how long a cost evaluation, a linearization and a solve take. `reduction` and
`storage` are what `:auto` builds; the other three choices follow the rules in its docstring. Every threshold in them is a
starting value, not yet measured.

The result, an [`OptimizationRecommendation`](@ref), explains itself when displayed. For a 2D pose chain of 30 poses with
40 eliminable landmarks, at its starting values:

```
julia> rec = recommendoptimizer(model)
OptimizationRecommendation for 70 unfixed variables of total dimension 170, 130
costs and 290 residual rows, at the model's current values:
  formulation = :normal
      Every cost is a residual, and the condition estimate of the Jacobi-scaled
      Hessian, 118000.0, is below 1.0e12: the normal equations are accurate
      enough.
  reduction   = :eliminate
      What reduction = :auto builds on the Hessian path: 80 of the 170
      dimensions are eliminated (the Schur complement), leaving a reduced system
      of 90.
  storage     = :sparse
      What storage = :auto builds: the 90×90 reduced system is stored sparse,
      from the predicted fill of its Cholesky factor.
  linsolver   = :cholesky
      Cholesky of the reduced system (falling back to Bunch-Kaufman), which
      linsolver = :auto uses for the Schur complement.
  iterator    = :dogleg
      The undamped solve succeeded, the condition estimate of the Jacobi-scaled
      Hessian, 118000.0, is below 1.0e8, the diagonal's spread (largest /
      smallest), 38.7, is below 1.0e6, and a solve costs at least a
      linearization: dogleg, which does not solve again after a rejected step.
Measured once each, after a warm-up call: a cost evaluation 3.71 μs, a
linearization 12.8 μs, an undamped solve 43.4 μs. The thresholds are starting
values, not measured.

The same choices, by hand (with SecondOrderOptimization loaded):
  optimize!(model,
      OptimizationOptions(iterator = :dogleg,
          linsolver = :cholesky);
      formulation = :normal, reduction = :eliminate, storage = :sparse)
```

There are two ways to use it, with `using SecondOrderOptimization`:

- `optimize!(model, rec)` runs the recommendation. Other keywords (`maxiters`, `maxtime`, the tolerances) go to
  `OptimizationOptions`; a keyword the recommendation sets is an `ArgumentError`.
- Copy the printed call and change it as needed. The options are then fixed in the code, which also saves the one
  dynamic dispatch of building them from the recommendation's values.

A recommendation holds for the model and the values it was made from: it is not checked again when it is used.
