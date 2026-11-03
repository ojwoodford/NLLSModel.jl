# Defining Residuals & Variables

In `NLLSModels.jl`, least-squares factors are modeled by subtypes of [`AbstractResidual`](@ref), and general scalar factors by subtypes of [`AbstractCost`](@ref).

## Defining an `AbstractResidual`

Every [`AbstractResidual`](@ref) subtype `R` defines:
1. [`NLLSModels.ndeps(::R)`](@ref ndeps): Number of variable blocks the residual depends on, returned as a `StaticInt` (`static(N)`), up to `MAX_ARGS` (10).
2. [`NLLSModels.nres(::R)`](@ref nres): Length of the residual vector, returned as `static(M)` (for static residuals) or an `Int` (for dynamic-length residuals).
3. [`NLLSModels.varindices(r::R)`](@ref varindices): Indices in `problem.variables` of the variables `r` depends on.
4. [`NLLSModels.getvars(r::R, vars::Vector)`](@ref getvars): Returns a concretely typed `Tuple` of the variable blocks for `r`.
5. [`NLLSModels.computeresidual(r::R, vars...)`](@ref computeresidual): Computes and returns the residual vector `SVector{M, T}` (or `Vector{T}` / scalar).
6. `Base.eltype(::R)`: Scalar element type (e.g. `Float64`).

### Example: Two-Variable Relative Pose Residual

```julia
using NLLSModels
using Static, StaticArrays

struct RelativePoseResidual <: AbstractResidual
    connection::SVector{2, Int}
    measurement::SVector{2, Float64}
end

NLLSModels.ndeps(::RelativePoseResidual) = static(2)
NLLSModels.nres(::RelativePoseResidual) = static(2)
NLLSModels.varindices(r::RelativePoseResidual) = r.connection
NLLSModels.getvars(r::RelativePoseResidual, vars::Vector) = (
    vars[r.connection[1]]::SVector{2, Float64},
    vars[r.connection[2]]::SVector{2, Float64},
)
NLLSModels.computeresidual(r::RelativePoseResidual, pose1, pose2) = (pose2 - pose1) - r.measurement
Base.eltype(::RelativePoseResidual) = Float64
```

## Automatic Differentiation vs Analytic Jacobians

By default, `NLLSModels.jl` computes exact Jacobians via `ForwardDiff.jl` using [`computeresidual`](@ref).

To supply an analytic Jacobian, specialize [`NLLSModels.computeresjac(varflags::StaticInt, r::R, vars...)`](@ref computeresjac), where bit `i` of `varflags` is set if variable `i` is unfixed:

```julia
function NLLSModels.computeresjac(::StaticInt{3}, r::RelativePoseResidual, pose1, pose2)
    res = (pose2 - pose1) - r.measurement
    # Columns correspond to pose1 (2 DoF) followed by pose2 (2 DoF)
    jac = @SMatrix [-1.0  0.0  1.0  0.0;
                     0.0 -1.0  0.0  1.0]
    return res, jac
end
```
