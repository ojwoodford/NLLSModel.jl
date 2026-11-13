# Defining Residuals & Variables

In `NLLSModel.jl`, least-squares factors are modeled by subtypes of [`AbstractResidual`](@ref), and general scalar factors by subtypes of [`AbstractCost`](@ref).

## Defining an `AbstractResidual`

Every [`AbstractResidual`](@ref) subtype `R` defines:
1. [`NLLSModel.ndeps(::R)`](@ref ndeps): Number of variable blocks the residual depends on, returned as a `StaticInt` (`static(N)`), up to `MAX_ARGS` (10).
2. [`NLLSModel.nres(::R)`](@ref nres): Length of the residual vector, returned as `static(M)` (for static residuals) or an `Int` (for dynamic-length residuals).
3. [`NLLSModel.varindices(r::R)`](@ref varindices): Indices in `problem.variables` of the variables `r` depends on.
4. [`NLLSModel.getvars(r::R, vars::Vector)`](@ref getvars): Returns a concretely typed `Tuple` of the variable blocks for `r`.
5. [`NLLSModel.computeresidual(r::R, vars...)`](@ref computeresidual): Computes and returns the residual vector `SVector{M, T}` (or `Vector{T}` / scalar).
6. `Base.eltype(::R)`: Scalar element type (e.g. `Float64`).

### Example: Two-Variable Relative Pose Residual

```julia
using NLLSModel
using Static, StaticArrays

struct RelativePoseResidual <: AbstractResidual
    connection::SVector{2, Int}
    measurement::SVector{2, Float64}
end

NLLSModel.ndeps(::RelativePoseResidual) = static(2)
NLLSModel.nres(::RelativePoseResidual) = static(2)
NLLSModel.varindices(r::RelativePoseResidual) = r.connection
NLLSModel.getvars(r::RelativePoseResidual, vars::Vector) = (
    vars[r.connection[1]]::SVector{2, Float64},
    vars[r.connection[2]]::SVector{2, Float64},
)
NLLSModel.computeresidual(r::RelativePoseResidual, pose1, pose2) = (pose2 - pose1) - r.measurement
Base.eltype(::RelativePoseResidual) = Float64
```

## Automatic Differentiation vs Analytic Jacobians

By default, `NLLSModel.jl` computes exact Jacobians via `ForwardDiff.jl` using [`computeresidual`](@ref).

To supply an analytic Jacobian, specialize [`NLLSModel.computeresjac(varflags::StaticInt, r::R, vars...)`](@ref computeresjac), where bit `i` of `varflags` is set if variable `i` is unfixed:

```julia
function NLLSModel.computeresjac(::StaticInt{3}, r::RelativePoseResidual, pose1, pose2)
    res = (pose2 - pose1) - r.measurement
    # Columns correspond to pose1 (2 DoF) followed by pose2 (2 DoF)
    jac = @SMatrix [-1.0  0.0  1.0  0.0;
                     0.0 -1.0  0.0  1.0]
    return res, jac
end
```

## Performance: static and run-time sizes

Return a `StaticInt` from [`nres`](@ref) and `nvars` to make sizes static. The gradient and Hessian of each cost are
formed by one of three kernels, chosen at compile time: the full static multiply (static residual length `M` and
unfixed variable sizes totalling `n <= MAX_STATIC_VAR` with `M * n <= MAX_STATIC_JAC`), the static-column kernel
(otherwise, with a static `M <= MAX_STATIC_ROWS`), and the `@turbo` kernel (everything else). The three constants are
starting values (64, 128 and 16), not tuned.

Everything computed for one cost is single-threaded by design; the outer loop over costs is what is threaded. So your
`computeresidual`, [`computeresjac`](@ref) and [`computecostgradhess`](@ref) must be single-threaded (no BLAS on large
matrices, no `Threads.@threads`; use `BLAS.set_num_threads(1)` if you need BLAS), and arrays they return must not be
shared with other residuals. See the "Performance" section of the README for the table of kernels.
