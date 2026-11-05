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
