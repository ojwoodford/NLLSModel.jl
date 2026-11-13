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

## Eliminating Variables: `reduction` and `reduced`

Whether eliminable variables (`iseliminable`, for example bundle-adjustment landmarks) are eliminated by a Schur complement is a
choice about the layout of the linear system, so it is made where the system is built: by keywords of
`optimize!(model, options; reduction, reduced)` and of [`optimvars`](@ref), not in `OptimizationOptions`. The optimizer
eliminates exactly when the Hessian prototype it is given is a `SchurBlockMatrix`. `linsolver` names only the solver for
the system that is left (the reduced system, after elimination).

- `reduction = :auto` (default) eliminates when the model has eliminable variables and the structure fits (each eliminable
  variable of dimension 1 to 3, coupled only to retained variables), and falls back to no elimination otherwise.
  `:eliminate` eliminates or throws an `ArgumentError`. `:none` never eliminates. Other values are an `ArgumentError`.
- `reduced = :auto` (default) stores the reduced system dense or sparse from the predicted fill of its Cholesky factor;
  `:dense` and `:sparse` force it. It applies only when elimination happens: with `reduction = :none`, or `:auto` on a model
  that does not eliminate, it is silently ignored (no error, no warning).

```julia
optimize!(problem, options; reduction = :eliminate, reduced = :sparse)
```
