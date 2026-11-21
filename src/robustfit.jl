# Fitting an adaptive kernel to squared residual norms, by expectation-maximization: on its own (`optimizerobustifier`)
# and to the residuals of a model that use it (`optimizerobustifier!`)

# `wprior = (α, β)`: a Beta prior on the first component's weight, as pseudo-counts, both finite and at least 1
function checkwprior(wprior)
    α, β = wprior
    (isfinite(α) && isfinite(β) && α >= 1 && β >= 1) ||
        throw(ArgumentError("wprior = (α, β) must have α and β finite and at least 1 (got wprior = $wprior); (1, 1) is no prior"))
    return (Float64(α), Float64(β))
end

function optimizerobustifier(kernel::Union{ContaminatedGaussian, GaussianPlusUniform}, s::AbstractVector{<:Real};
        maxiters::Integer=10, rtol::Real=1e-6, wprior=(1.0, 1.0))
    return emfit(kernel, s, nothing, maxiters, rtol, checkwprior(wprior))
end
function optimizerobustifier(kernel::GaussianPlusUniform, s::AbstractVector{<:Real}, logvolume::AbstractVector{<:Real};
        maxiters::Integer=10, rtol::Real=1e-6, wprior=(1.0, 1.0))
    prior = checkwprior(wprior)
    length(logvolume) == length(s) ||
        throw(DimensionMismatch("logvolume has length $(length(logvolume)) and s has length $(length(s)); they must be equal"))
    return emfit(kernel, s, logvolume, maxiters, rtol, prior)
end

# The log volume of the outlier term of the `j`-th residual: its own, or the kernel's
@inline residuallogvolume(::Nothing, kernel, j) = kernel.logvolume
@inline residuallogvolume(logvolume, kernel, j) = @inbounds logvolume[firstindex(logvolume) + j - 1]

# The M step of the first component's weight, with the prior's pseudo-counts
@inline mstepweight(sumgamma, n, (α, β)) = (sumgamma + α - 1) / (n + α + β - 2)

function emfit(kernel::GaussianPlusUniform{T, D}, s, logvolume, maxiters, rtol, prior) where {T, D}
    n = length(s)
    n == 0 && return kernel
    oldparams = params(kernel)
    for _ in 1:maxiters
        # E step: each residual's inlier responsibility. M step: the totals.
        sumgamma = zero(T)
        sumgammas = zero(T)
        for (j, si) in enumerate(s)
            gamma = responsibility((kernel.k - si * kernel.halfinvvar) - (kernel.log1mw - residuallogvolume(logvolume, kernel, j)))
            sumgamma += gamma
            sumgammas += gamma * si
        end
        0 < sumgamma < n || break
        variance = sumgammas / (D * sumgamma)
        (isfinite(variance) && variance > 0) || break
        kernel = GaussianPlusUniform(ZeroToInfScalar{T}(inv(sqrt(variance))), ZeroToOneScalar{T}(T(mstepweight(sumgamma, n, prior))),
            static(D), kernel.logvolume)
        newparams = params(kernel)
        isapprox(oldparams, newparams; rtol) && break
        oldparams = newparams
    end
    return kernel
end

function emfit(kernel::ContaminatedGaussian{T, D}, s, ::Nothing, maxiters, rtol, prior) where {T, D}
    n = length(s)
    n == 0 && return kernel
    oldparams = params(kernel)
    for _ in 1:maxiters
        # E step: each residual's responsibility of the first component. M step: the totals.
        sumgamma = zero(T)
        sumgammas = zero(T)
        sumrests = zero(T)
        for si in s
            gamma = responsibility((kernel.k1 - si * kernel.halfinvvar1) - (kernel.k2 - si * kernel.halfinvvar2))
            sumgamma += gamma
            sumgammas += gamma * si
            sumrests += (1 - gamma) * si
        end
        0 < sumgamma < n || break
        variance1 = sumgammas / (D * sumgamma)
        variance2 = sumrests / (D * (n - sumgamma))
        (isfinite(variance1) && variance1 > 0 && isfinite(variance2) && variance2 > 0) || break
        kernel = ContaminatedGaussian(ZeroToInfScalar{T}(inv(sqrt(variance1))), ZeroToInfScalar{T}(inv(sqrt(variance2))),
            ZeroToOneScalar{T}(T(mstepweight(sumgamma, n, prior))), static(D))
        newparams = params(kernel)
        isapprox(oldparams, newparams; rtol) && break
        oldparams = newparams
    end
    return kernel
end

function optimizerobustifier!(model::Model, index::Integer; s::AbstractVector{Float64}=Float64[],
        logvolume::AbstractVector{Float64}=Float64[], maxiters::Integer=10, rtol::Real=1e-6, wprior=(1.0, 1.0))
    kernel = model.variables[index]
    kernel isa Union{ContaminatedGaussian, GaussianPlusUniform} ||
        throw(ArgumentError("optimizerobustifier!: model.variables[$index] is a $(typeof(kernel)); it must be a ContaminatedGaussian or a GaussianPlusUniform"))
    prior = checkwprior(wprior)
    fitmodelkernel!(model, Int(index), kernel, s, logvolume, maxiters, rtol, prior)
    return cost(model)
end

# A function barrier on the kernel's type: count the residuals that use the kernel, collect their squared norms (and,
# for GaussianPlusUniform, the log volume of the kernel each hands over), fit, and write the kernel back
function fitmodelkernel!(model, index, kernel::K, s, logvolume, maxiters, rtol, prior) where K
    costs = values(model.costs)
    n = countkernelresiduals(index, costs...)
    n == 0 && return nothing
    resize!(s, n)
    K <: GaussianPlusUniform && resize!(logvolume, n)
    collectkernelresiduals!(s, logvolume, K, model.variables, index, 0, costs...)
    model.variables[index] = K <: GaussianPlusUniform ? emfit(kernel, s, logvolume, maxiters, rtol, prior) :
                                                        emfit(kernel, s, nothing, maxiters, rtol, prior)
    return nothing
end

# The adaptive residuals whose kernel is variable `index`, over the model's cost vectors (one type each)
countkernelresiduals(index) = 0
countkernelresiduals(index, v::AbstractVector, rest...) = countkernelresidualsvec(index, v) + countkernelresiduals(index, rest...)
countkernelresidualsvec(index, v::AbstractVector) = 0
function countkernelresidualsvec(index, v::AbstractVector{<:AbstractAdaptiveResidual})
    n = 0
    for c in v
        n += varindices(c)[1] == index
    end
    return n
end

collectkernelresiduals!(s, logvolume, K, vars, index, j) = j
function collectkernelresiduals!(s, logvolume, K, vars, index, j, v::AbstractVector, rest...)
    j = collectkernelresidualsvec!(s, logvolume, K, vars, index, j, v)
    return collectkernelresiduals!(s, logvolume, K, vars, index, j, rest...)
end
collectkernelresidualsvec!(s, logvolume, K, vars, index, j, v::AbstractVector) = j
function collectkernelresidualsvec!(s, logvolume, K, vars, index, j, v::AbstractVector{<:AbstractAdaptiveResidual})
    for c in v
        varindices(c)[1] == index || continue
        cvars = getvars(c, vars)
        j += 1
        @inbounds s[j] = Float64(sqnorm(computeresidual(c, Base.tail(cvars)...)))
        storelogvolume!(logvolume, K, j, cvars[1])
    end
    return j
end

# The log volume of the kernel a residual hands over (GaussianPlusUniform only)
@inline storelogvolume!(logvolume, ::Type{<:GaussianPlusUniform}, j, kernel) = (@inbounds logvolume[j] = kernel.logvolume; nothing)
@inline storelogvolume!(logvolume, ::Type, j, kernel) = nothing
