const NDIMS = 3
struct LinearResidualStatic <: NLLSModel.AbstractResidual
    y::SVector{NDIMS, Float64}
    X::SMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS}
    varind::Int
end
NLLSModel.ndeps(::LinearResidualStatic) = static(1)
NLLSModel.nres(::LinearResidualStatic) = static(NDIMS)
NLLSModel.varindices(res::LinearResidualStatic) = res.varind
NLLSModel.getvars(res::LinearResidualStatic, vars::Vector) = (vars[res.varind]::SVector{NDIMS, Float64},)
NLLSModel.computeresidual(res::LinearResidualStatic, w) = res.X * w - res.y
Base.eltype(::LinearResidualStatic) = Float64

struct LinearResidualDynamic <: NLLSModel.AbstractResidual
    y::Vector{Float64}
    X::Matrix{Float64}
    varind::Int
end
NLLSModel.ndeps(::LinearResidualDynamic) = static(1)
NLLSModel.nres(res::LinearResidualDynamic) = length(res.y)
NLLSModel.varindices(res::LinearResidualDynamic) = res.varind
NLLSModel.getvars(res::LinearResidualDynamic, vars::Vector) = (vars[res.varind]::Vector{Float64},)
NLLSModel.computeresidual(res::LinearResidualDynamic, w) = res.X * w - res.y
Base.eltype(::LinearResidualDynamic) = Float64

struct LinearCostStatic <: NLLSModel.AbstractCost
    y::SVector{NDIMS, Float64}
    varind::Int
end
NLLSModel.ndeps(::LinearCostStatic) = static(1)
NLLSModel.varindices(cost::LinearCostStatic) = cost.varind
NLLSModel.getvars(cost::LinearCostStatic, vars::Vector) = (vars[cost.varind]::SVector{NDIMS, Float64},)
NLLSModel.computecost(cost::LinearCostStatic, w) = cost.y' * w
Base.eltype(::LinearCostStatic) = Float64

struct LinearCostDynamic <: NLLSModel.AbstractCost
    y::Vector{Float64}
    varind::Int
end
NLLSModel.ndeps(::LinearCostDynamic) = static(1)
NLLSModel.varindices(cost::LinearCostDynamic) = cost.varind
NLLSModel.getvars(cost::LinearCostDynamic, vars::Vector) = (vars[cost.varind]::Vector{Float64},)
NLLSModel.computecost(cost::LinearCostDynamic, w) = cost.y' * w
Base.eltype(::LinearCostDynamic) = Float64
