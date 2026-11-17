struct LinearResidual <: NLLSModel.AbstractResidual
    y::Float64
    X::Vector{Float64}
end
NLLSModel.ndeps(::LinearResidual) = static(1) # Residual depends on 1 variables
NLLSModel.nres(::LinearResidual) = static(1) # Residual has length 1
NLLSModel.varindices(::LinearResidual) = 1
NLLSModel.getvars(::LinearResidual, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(res::LinearResidual, w) = res.X' * w - res.y
Base.eltype(::LinearResidual) = Float64

struct NormResidual <: NLLSModel.AbstractResidual
    len::Int
end
NLLSModel.ndeps(::NormResidual) = static(1) # Residual depends on 1 variables
NLLSModel.nres(res::NormResidual) = res.len
NLLSModel.varindices(::NormResidual) = 1
NLLSModel.getvars(::NormResidual, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(::NormResidual, w) = w
Base.eltype(::NormResidual) = Float64
