using Test
using NLLSModel
using NLPModels
using Static, StaticArrays, SparseArrays, LinearAlgebra
import OptimizationVariableInterface: nvars, update, update!, syncvars!, ZeroToInfScalar
import JSOSolvers
import LeastSquaresOptim

struct TestRosenbrock1 <: AbstractResidual end
NLLSModel.ndeps(::TestRosenbrock1) = static(1)
NLLSModel.nres(::TestRosenbrock1) = static(1)
NLLSModel.varindices(::TestRosenbrock1) = SVector(1)
NLLSModel.getvars(::TestRosenbrock1, vars::AbstractVector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::TestRosenbrock1, x) = SVector(1.0 - x[1])
Base.eltype(::TestRosenbrock1) = Float64

struct TestRosenbrock2 <: AbstractResidual end
NLLSModel.ndeps(::TestRosenbrock2) = static(1)
NLLSModel.nres(::TestRosenbrock2) = static(1)
NLLSModel.varindices(::TestRosenbrock2) = SVector(1)
NLLSModel.getvars(::TestRosenbrock2, vars::AbstractVector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::TestRosenbrock2, x) = SVector(10.0 * (x[2] - x[1]^2))
Base.eltype(::TestRosenbrock2) = Float64

# Multi-dimensional residual (nres = 2)
struct MultiDimRes <: AbstractResidual end
NLLSModel.ndeps(::MultiDimRes) = static(1)
NLLSModel.nres(::MultiDimRes) = static(2)
NLLSModel.varindices(::MultiDimRes) = SVector(1)
NLLSModel.getvars(::MultiDimRes, vars::AbstractVector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::MultiDimRes, x) = SVector(x[1] - 2.0, 2.0 * x[2] - 4.0)
Base.eltype(::MultiDimRes) = Float64

# Scalar varindices (ndeps = 1, returning Int)
struct ScalarVarIndRes <: AbstractResidual end
NLLSModel.ndeps(::ScalarVarIndRes) = static(1)
NLLSModel.nres(::ScalarVarIndRes) = static(1)
NLLSModel.varindices(::ScalarVarIndRes) = 1
NLLSModel.getvars(::ScalarVarIndRes, vars::AbstractVector) = (vars[1]::Float64,)
NLLSModel.computeresidual(::ScalarVarIndRes, x) = SVector(x - 3.0)
Base.eltype(::ScalarVarIndRes) = Float64

# Pair difference residual
struct TestPairDiff <: AbstractResidual
    i::Int
    j::Int
    val::Float64
end
TestPairDiff(i, j) = TestPairDiff(i, j, 1.0)
NLLSModel.ndeps(::TestPairDiff) = static(2)
NLLSModel.nres(::TestPairDiff) = static(1)
NLLSModel.varindices(r::TestPairDiff) = SVector(r.i, r.j)
NLLSModel.getvars(r::TestPairDiff, vars::AbstractVector) = (vars[r.i]::SVector{1, Float64}, vars[r.j]::SVector{1, Float64})
NLLSModel.computeresidual(r::TestPairDiff, xi, xj) = SVector(xi[1] - xj[1] - r.val)
Base.eltype(::TestPairDiff) = Float64

# Dynamic vector residual
struct DynamicRes <: AbstractResidual
    dim::Int
end
NLLSModel.ndeps(::DynamicRes) = static(1)
NLLSModel.nres(d::DynamicRes) = d.dim
NLLSModel.varindices(::DynamicRes) = SVector(1)
NLLSModel.getvars(::DynamicRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(d::DynamicRes, x) = x .- 2.0
Base.eltype(::DynamicRes) = Float64

# Non-finite producing residual
struct NonFiniteRes <: AbstractResidual
    mode::Symbol
end
NLLSModel.ndeps(::NonFiniteRes) = static(1)
NLLSModel.nres(::NonFiniteRes) = static(1)
NLLSModel.varindices(::NonFiniteRes) = SVector(1)
NLLSModel.getvars(::NonFiniteRes, vars::AbstractVector) = (vars[1]::SVector{1, Float64},)
function NLLSModel.computeresidual(r::NonFiniteRes, x)
    if r.mode === :overflow
        return SVector(1e200 * x[1])
    elseif r.mode === :nan
        return SVector(NaN * x[1])
    else
        return SVector(Inf * x[1])
    end
end
Base.eltype(::NonFiniteRes) = Float64

# Generic wrapper to attach any robustifier to any residual
# Variables for the flat-vector check: a Euclidean AbstractVector of our own (counting its updates), a rotation (an
# angle wrapped to (-π, π], so its update is not addition), and an AbstractVector that cannot be rebuilt from its values
const EUCLID_UPDATES = Ref(0)
struct EuclidVec <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(v::EuclidVec) = size(v.data)
Base.getindex(v::EuclidVec, i::Int) = v.data[i]
update(v::EuclidVec, δ, start=1) = (EUCLID_UPDATES[] += 1; EuclidVec(v.data .+ view(δ, start:start+length(v)-1)))
struct WrappedAngle <: AbstractVector{Float64}
    θ::Float64
end
WrappedAngle(x::AbstractVector) = WrappedAngle(x[1])
Base.size(::WrappedAngle) = (1,)
Base.getindex(v::WrappedAngle, i::Int) = v.θ
update(v::WrappedAngle, δ, start=1) = WrappedAngle(rem2pi(v.θ + δ[start], RoundNearest))
struct UnbuildableVec <: AbstractVector{Float64}
    data::Vector{Float64}
    tag::Int
end
Base.size(v::UnbuildableVec) = size(v.data)
Base.getindex(v::UnbuildableVec, i::Int) = v.data[i]
# A unit 4-vector with 3 degrees of freedom, stored in ambient coordinates: longer than its nvars (AN4)
struct AmbientVec <: AbstractVector{Float64}
    data::SVector{4, Float64}
end
AmbientVec(x::AbstractVector) = AmbientVec(normalize(SVector{4, Float64}(x)))
Base.size(::AmbientVec) = (4,)
Base.getindex(v::AmbientVec, i::Int) = v.data[i]
nvars(::AmbientVec) = static(3)
update(v::AmbientVec, δ, start=1) = AmbientVec(v.data + SVector(δ[start], δ[start+1], δ[start+2], 0.0))
struct EuclidRes <: AbstractResidual
    i::Int
end
NLLSModel.ndeps(::EuclidRes) = static(1)
NLLSModel.nres(::EuclidRes) = static(1)
NLLSModel.varindices(r::EuclidRes) = r.i
NLLSModel.getvars(r::EuclidRes, vars::AbstractVector) = (vars[r.i],)
NLLSModel.computeresidual(::EuclidRes, x) = SVector(sum(x))
Base.eltype(::EuclidRes) = Float64
struct AngleRes <: AbstractResidual
    i::Int
end
NLLSModel.ndeps(::AngleRes) = static(1)
NLLSModel.nres(::AngleRes) = static(1)
NLLSModel.varindices(r::AngleRes) = r.i
NLLSModel.getvars(r::AngleRes, vars::AbstractVector) = (vars[r.i]::WrappedAngle,)
NLLSModel.computeresidual(::AngleRes, x) = SVector(x.θ)
Base.eltype(::AngleRes) = Float64

struct RobustWrapper{R<:AbstractResidual, K<:AbstractRobustifier} <: AbstractResidual
    res::R
    kernel::K
end
NLLSModel.ndeps(r::RobustWrapper) = ndeps(r.res)
NLLSModel.nres(r::RobustWrapper) = nres(r.res)
NLLSModel.varindices(r::RobustWrapper) = varindices(r.res)
NLLSModel.getvars(r::RobustWrapper, vars::AbstractVector) = getvars(r.res, vars)
NLLSModel.computeresidual(r::RobustWrapper, args...) = computeresidual(r.res, args...)
NLLSModel.robustkernel(r::RobustWrapper) = r.kernel
Base.eltype(r::RobustWrapper) = eltype(r.res)

# Kernel for testing clamp and domain error
struct MockClampingKernel <: AbstractRobustifier
    target_dc::Float64
end
NLLSModel.robustify(k::MockClampingKernel, s) = s
NLLSModel.robustifydcost(k::MockClampingKernel, s) = (s, k.target_dc, 0.0)

struct ZeroWeightKernel <: AbstractRobustifier end
NLLSModel.robustify(::ZeroWeightKernel, s) = 15.0
NLLSModel.robustifydcost(::ZeroWeightKernel, s) = (15.0, 0.0, 0.0)

# Adaptive residual definition for test
struct TestAdaptiveRes <: AbstractAdaptiveResidual
    varind::SVector{2, Int}
end
NLLSModel.ndeps(::TestAdaptiveRes) = static(2)
NLLSModel.nres(::TestAdaptiveRes) = static(1)
NLLSModel.varindices(r::TestAdaptiveRes) = r.varind
NLLSModel.getvars(r::TestAdaptiveRes, vars::Vector) = (vars[r.varind[1]]::ContaminatedGaussian{Float64}, vars[r.varind[2]]::Float64)
NLLSModel.computeresidual(::TestAdaptiveRes, x) = SVector(x - 1.0)
Base.eltype(::TestAdaptiveRes) = Float64

# Finite difference helper
function finite_diff_jac(res!, x, M; h=1e-7)
    N = length(x)
    J = Matrix{Float64}(undef, M, N)
    r_plus = Vector{Float64}(undef, M)
    r_minus = Vector{Float64}(undef, M)
    x_pert = copy(x)
    for j in 1:N
        x_pert[j] = x[j] + h
        res!(r_plus, x_pert)
        x_pert[j] = x[j] - h
        res!(r_minus, x_pert)
        x_pert[j] = x[j]
        J[:, j] .= (r_plus .- r_minus) ./ (2 * h)
    end
    return J
end

# A residual that counts its evaluations (AN3)
const NLS_EVALS = Ref(0)
struct CountedNLS{C} <: AbstractResidual
    c::C
end
NLLSModel.ndeps(r::CountedNLS) = NLLSModel.ndeps(r.c)
NLLSModel.nres(r::CountedNLS) = NLLSModel.nres(r.c)
NLLSModel.varindices(r::CountedNLS) = NLLSModel.varindices(r.c)
NLLSModel.getvars(r::CountedNLS, vars::AbstractVector) = NLLSModel.getvars(r.c, vars)
NLLSModel.computeresidual(r::CountedNLS, args...) = (NLS_EVALS[] += 1; NLLSModel.computeresidual(r.c, args...))
Base.eltype(::CountedNLS) = Float64
