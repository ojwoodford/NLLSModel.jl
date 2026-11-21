using NLLSModel, Test, Static, StaticArrays, Random
using OptimizationVariableInterface: update!, update

struct SimpleResidual <: NLLSModel.AbstractAdaptiveResidual
    data::Float64
    varind::Int
end
NLLSModel.ndeps(::SimpleResidual) = static(2)
NLLSModel.nres(::SimpleResidual) = static(1)
NLLSModel.varindices(res::SimpleResidual) = SVector(1, res.varind)
NLLSModel.getvars(res::SimpleResidual, vars::Vector) = (vars[1]::NLLSModel.ContaminatedGaussian{Float64, 1}, vars[res.varind]::Float64)
NLLSModel.computeresidual(res::SimpleResidual, mean) = mean - res.data
NLLSModel.computeresjac(varflags, res::SimpleResidual, mean) = mean - res.data, SVector(one(mean))'
Base.eltype(::SimpleResidual) = Float64
