# Robust Cost Functions

Real-world sensor measurements and feature correspondences frequently contain outliers. Standard squared loss ($\frac{1}{2}\|r\|^2$) allows large outliers to pull estimated parameters far away from ground truth.

`NLLSModel.jl` provides robust M-estimator kernels that transform the squared residual norm $s = \|r(x)\|^2$ into $\frac{1}{2} \rho(s)$.

## Available Robustifiers

| Robustifier | Formula $\rho(s)$ (where $s = \|r\|^2$) | Description |
| :--- | :--- | :--- |
| [`NoRobust()`](@ref) | $s$ | Standard unrobustified squared loss (default) |
| [`Scaled(kernel, h)`](@ref) | $h \cdot \rho(s)$ | Multiplies another kernel by constant weight $h$ |
| [`HuberKernel(w)`](@ref) | $s$ if $s < w^2$, else $2 w \sqrt{s} - w^2$ | First-order Huber loss with linear tails |
| [`Huber2oKernel(w)`](@ref) | $s$ if $s < w^2$, else $2 w \sqrt{s} - w^2$ | Huber loss including second-order derivative $\rho''(s)$ |
| [`GemanMcclureKernel(w)`](@ref) | $\frac{s w^2}{s + w^2}$ | First-order Geman-McClure truncated loss |
| [`GemanMcclure2oKernel(w)`](@ref) | $\frac{s w^2}{s + w^2}$ | Geman-McClure loss including second-order derivative $\rho''(s)$ |
| [`ContaminatedGaussian(s1, s2, w, d)`](@ref) | $-2\log\left(w\,N_d(s; s_1) + (1-w)\,N_d(s; s_2)\right)$ | Adaptive mixture of two Gaussians |
| [`GaussianPlusUniform(σ, w, d; volume = A)`](@ref) | $-2\log\left(w\,N_d(s; \sigma) + (1-w)/A\right)$ | Adaptive Gaussian plus uniform outliers |

Here $N_d(s; \sigma) = (2\pi\sigma^2)^{-d/2} e^{-s/(2\sigma^2)}$ is the density of a zero-mean Gaussian in $d$ dimensions,
at a point of squared norm $s$. A fixed kernel may drop a constant; an adaptive kernel is $\rho(s) = -2 \log p(r)$ exactly,
for a density $p$ over the residual $r$, so its cost can be negative.

## Attaching a Robustifier to a Residual

To use a robust kernel with a residual type, store or return the kernel from [`NLLSModel.robustkernel`](@ref):

```julia
using NLLSModel
using Static, StaticArrays

struct RobustPointResidual{K <: AbstractRobustifier} <: AbstractResidual
    target::SVector{2, Float64}
    varind::Int
    kernel::K
end

NLLSModel.ndeps(::RobustPointResidual) = static(1)
NLLSModel.nres(::RobustPointResidual) = static(2)
NLLSModel.varindices(r::RobustPointResidual) = SVector(r.varind)
NLLSModel.getvars(r::RobustPointResidual, vars::Vector) = (vars[r.varind]::SVector{2, Float64},)
NLLSModel.computeresidual(r::RobustPointResidual, x) = x - r.target
NLLSModel.robustkernel(r::RobustPointResidual) = r.kernel
Base.eltype(::RobustPointResidual) = Float64

problem = Model(SVector{2, Float64}, RobustPointResidual{Huber2oKernel{Float64}})
idx = addvariable!(problem, SVector(0.0, 0.0))
addcost!(problem, RobustPointResidual(SVector(1.0, 1.0), idx, Huber2oKernel(1.5)))
addcost!(problem, RobustPointResidual(SVector(1.1, 0.9), idx, Huber2oKernel(1.5)))
addcost!(problem, RobustPointResidual(SVector(50.0, -40.0), idx, Huber2oKernel(1.5))) # Outlier

res = optimize!(problem) # Needs `using SecondOrderOptimization`
```

## Robust Kernels in the Jacobian Formulation

The residual-and-Jacobian evaluators ([`optimvars_nls`](@ref), [`flat_optimvars_nls`](@ref)) have no robust mode. As on
the Hessian path, the kernel of each cost decides how its rows enter, from what `robustifydcost` returns:

| Kernel | Rows of that cost |
|:---|:---|
| linear (`NoRobust`, a valid `Scaled`) | unweighted or scaled by a constant |
| second derivative zero (`HuberKernel`, `GemanMcclureKernel`) | IRLS: scaled by the square root of the first derivative |
| second derivative non-zero (`Huber2oKernel`, `GemanMcclure2oKernel`) | IRLS with the Triggs correction |

Choosing first- or second-order treatment therefore means choosing the kernel type. `resjac!` returns the true cost,
which equals `½‖r‖²` only when every kernel is linear; [`islinear`](@ref)`(resjac!)` reports that, and
[`NLLSNLSModel`](@ref) (which hands `r` to solvers that derive the cost from it) requires it.

## Adaptive Kernels

An adaptive kernel ([`ContaminatedGaussian`](@ref), [`GaussianPlusUniform`](@ref)) has parameters of its own, the scales
and weight of its components. It is a variable of the model, and the first variable of each
[`AbstractAdaptiveResidual`](@ref) that uses it. Its constructor takes `d`, the length of those residuals, as
`static(d)`, because the density's normalization depends on it; [`addcost!`](@ref) checks it against `nres`.

The example below estimates a 2-vector mean from 500 measurements, 80 % of them within unit noise of `(1, −1)` and the
rest anywhere in a 40 × 40 square.

```julia
using NLLSModel, Static, StaticArrays, Random
using SecondOrderOptimization

struct MeanResidual{K} <: AbstractAdaptiveResidual
    y::SVector{2, Float64}
end
NLLSModel.ndeps(::MeanResidual) = static(2)
NLLSModel.nres(::MeanResidual) = static(2)
NLLSModel.varindices(::MeanResidual) = SVector(1, 2)
NLLSModel.getvars(::MeanResidual{K}, vars::Vector) where K = (vars[1]::K, vars[2]::SVector{2, Float64})
NLLSModel.computeresidual(r::MeanResidual, x) = x - r.y
Base.eltype(::MeanResidual) = Float64

function meanmodel()
    kernel = GaussianPlusUniform(0.5, 0.5, static(2); volume = 40.0^2) # outliers anywhere in a 40 × 40 square
    K = typeof(kernel)
    model = Model(Union{K, SVector{2, Float64}}, MeanResidual{K})
    addvariable!(model, kernel)            # variable 1: the kernel
    addvariable!(model, SVector(0.0, 0.0)) # variable 2: the mean
    rng = MersenneTwister(1)
    for _ in 1:500
        inlier = rand(rng) < 0.8
        y = inlier ? SVector(1.0, -1.0) + randn(rng, SVector{2, Float64}) : 40 .* (rand(rng, SVector{2, Float64}) .- 0.5)
        addcost!(model, MeanResidual{K}(y))
    end
    return model
end
```

**Joint optimization.** With the kernel unfixed, the optimizer adjusts its parameters with the other variables. This is
supported on the Hessian path (the default) only:

```julia
model = meanmodel()
optimize!(model)
println(model.variables[2]) # [0.993603791286982, -0.9505227226169588]
```

**Alternation.** Fix the kernel, and refit it to the residuals in the optimizer's callback with
[`optimizerobustifier!`](@ref), which returns the model's new cost. A callback that changes the cost needs
`callbackchangesstate = true`. This works on both paths; it is the only way to adapt a kernel with
`formulation = :jacobian`, which does not take an unfixed adaptive kernel. The buffers `s` and `logvolume`, kept between
calls, mean the fit allocates nothing after the first.

```julia
model = meanmodel()
s, logvolume = Float64[], Float64[]
callback = (cost, x, data, iteratedata) -> (optimizerobustifier!(model, 1; s, logvolume), 0)
optimize!(model, OptimizationOptions(callbackchangesstate = true); unfixed = 2, callback, formulation = :jacobian)
println(model.variables[2]) # [0.9936037935498518, -0.9505227122229205]
```

[`optimizerobustifier`](@ref) fits a kernel to a vector of squared residual norms directly.

**The outlier volume.** `GaussianPlusUniform`'s `volume`, `A`, is the volume of the region the outliers can fall in,
in the units of the residual. It is supplied, not optimized, and need not be accurate. When residuals differ, each can
hand over a volume of its own: return [`withlogvolume`](@ref)`(kernel, ℓᵢ)` from `getvars` (an adaptive kernel) or
from `robustkernel` (a fixed one), with `ℓᵢ = log Aᵢ`. For a residual whitened by a covariance `Σᵢ` (`rᵢ = Lᵢ⁻¹ eᵢ`
with `Σᵢ = Lᵢ Lᵢᵀ`), a region of volume `A` in the units of `eᵢ` has volume `A / √det Σᵢ`, so
`ℓᵢ = log A − ½ log det Σᵢ`. [`optimizerobustifier!`](@ref) reads each residual's `ℓᵢ` from the kernel its `getvars`
hands over.

**Which kernel.** Use `GaussianPlusUniform` where the volume of the outlier region is known, and `ContaminatedGaussian`
where it is not: its second Gaussian, whose scale is fitted, takes the place of the uniform term.
