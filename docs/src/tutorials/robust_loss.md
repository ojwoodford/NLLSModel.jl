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
| [`ContaminatedGaussian(s1, s2, w)`](@ref) | $-\log\left(w e^{-s/(2s_1^2)}/s_1 + (1-w) e^{-s/(2s_2^2)}/s_2\right)$ | Adaptive inlier/outlier Gaussian mixture kernel |

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

res = optimize!(problem)
```
