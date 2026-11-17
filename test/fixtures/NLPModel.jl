# Define sample residuals for tests
struct Rosenbrock1 <: AbstractResidual end
NLLSModel.ndeps(::Rosenbrock1) = static(1)
NLLSModel.nres(::Rosenbrock1) = static(1)
NLLSModel.varindices(::Rosenbrock1) = SVector(1)
NLLSModel.getvars(::Rosenbrock1, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::Rosenbrock1, x) = SVector(1.0 - x[1])
Base.eltype(::Rosenbrock1) = Float64

struct Rosenbrock2 <: AbstractResidual end
NLLSModel.ndeps(::Rosenbrock2) = static(1)
NLLSModel.nres(::Rosenbrock2) = static(1)
NLLSModel.varindices(::Rosenbrock2) = SVector(1)
NLLSModel.getvars(::Rosenbrock2, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::Rosenbrock2, x) = SVector(10.0 * (x[2] - x[1]^2))
Base.eltype(::Rosenbrock2) = Float64

struct PairDiff <: AbstractResidual
    i::Int
    j::Int
    val::Float64
end
PairDiff(i, j) = PairDiff(i, j, 1.0)
NLLSModel.ndeps(::PairDiff) = static(2)
NLLSModel.nres(::PairDiff) = static(1)
NLLSModel.varindices(r::PairDiff) = SVector(r.i, r.j)
NLLSModel.getvars(r::PairDiff, vars::Vector) = (vars[r.i]::SVector{1, Float64}, vars[r.j]::SVector{1, Float64})
NLLSModel.computeresidual(r::PairDiff, xi, xj) = SVector(xi[1] - xj[1] - r.val)
Base.eltype(::PairDiff) = Float64

struct AnchorVal <: AbstractResidual
    val::Float64
end
AnchorVal() = AnchorVal(0.0)
NLLSModel.ndeps(::AnchorVal) = static(1)
NLLSModel.nres(::AnchorVal) = static(1)
NLLSModel.varindices(::AnchorVal) = SVector(1)
NLLSModel.getvars(::AnchorVal, vars::Vector) = (vars[1]::SVector{1, Float64},)
NLLSModel.computeresidual(r::AnchorVal, x1) = SVector(x1[1] - r.val)
Base.eltype(::AnchorVal) = Float64

# Mark 2-vectors as eliminable (test-only; nothing else in the tests uses a 2-vector as a variable)
OptimizationVariableInterface.iseliminable(::Type{<:SVector{2, Float64}}) = static(true)
NLLSModel.generatemeasurement(camera::SVector{6, T}, pt::SVector{2, U}) where {T, U} = SVector(camera[1] * pt[1] + camera[2] * pt[2], camera[5] * pt[1] + camera[6] * pt[2])

# Variables NLPLSModel cannot take (AN9): a struct holding its coordinates, which does not pack into a flat vector (as a
# bundle-adjustment point type might), and a vector whose update is multiplicative, not addition
struct OpaquePoint
    p::SVector{3, Float64}
end
nvars(::OpaquePoint) = static(3)
OptimizationVariableInterface.update(v::OpaquePoint, δ, start=1) = OpaquePoint(v.p + SVector(δ[start], δ[start+1], δ[start+2]))
struct ScaleVec <: AbstractVector{Float64}
    data::SVector{2, Float64}
end
ScaleVec(x::AbstractVector) = ScaleVec(SVector{2, Float64}(x))
Base.size(::ScaleVec) = (2,)
Base.getindex(v::ScaleVec, i::Int) = v.data[i]
nvars(::ScaleVec) = static(2)
OptimizationVariableInterface.update(v::ScaleVec, δ, start=1) = ScaleVec(v.data .* exp.(SVector(δ[start], δ[start+1])))
# A residual on one variable of any type, counting its evaluations
const NLP_EVALS = Ref(0)
struct CountedPrior <: AbstractResidual
    i::Int
end
NLLSModel.ndeps(::CountedPrior) = static(1)
NLLSModel.nres(::CountedPrior) = static(1)
NLLSModel.varindices(r::CountedPrior) = SVector(r.i)
NLLSModel.getvars(r::CountedPrior, vars::Vector) = (vars[r.i],)
NLLSModel.computeresidual(::CountedPrior, x::OpaquePoint) = (NLP_EVALS[] += 1; SVector(sum(x.p) - 1.0))
NLLSModel.computeresidual(::CountedPrior, x) = (NLP_EVALS[] += 1; SVector(sum(x) - 1.0))
Base.eltype(::CountedPrior) = Float64
