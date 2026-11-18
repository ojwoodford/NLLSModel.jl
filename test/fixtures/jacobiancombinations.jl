using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra
using SecondOrderOptimization, Krylov
import BlockLinearSystems: SchurBlockJacobian

fixture("jacobianelimination")

# (reduction, storage) => (the Jacobian prototype, the linsolver values the optimizer accepts with it)
const TALL_TABLE = Dict(
    (:none, :dense) => (Matrix{Float64}, (:auto, :qr, :lsmr)),
    (:none, :sparse) => (SparseMatrixCSC{Float64, Int}, (:auto, :qr, :lsmr)),
    (:eliminate, :dense) => (SchurBlockJacobian{Float64, Matrix{Float64}}, (:auto, :qr, :cholesky, :ldl, :lsmr)),
    (:eliminate, :sparse) => (SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}, (:auto, :qr, :cholesky, :ldl, :lsmr)),
)
# Every iterator; every row takes each of them (gradient descent since SecondOrderOptimization IT2)
const TALL_ITERATORS = (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton, :gradientdescent)
# Every linsolver symbol an extension accepts
const ALL_LINSOLVERS = (:auto, :cholesky, :ldl, :qr, :cg, :minres, :krylov, :lsmr, :pardiso)

# A cost that counts its evaluations (the residual, and the residual under autodiff)
const TALL_EVALS = Threads.Atomic{Int}(0)
struct Counted{C} <: NLLSModel.AbstractResidual
    c::C
end
NLLSModel.ndeps(r::Counted) = NLLSModel.ndeps(r.c)
NLLSModel.nres(r::Counted) = NLLSModel.nres(r.c)
NLLSModel.varindices(r::Counted) = NLLSModel.varindices(r.c)
NLLSModel.getvars(r::Counted, vars::AbstractVector) = NLLSModel.getvars(r.c, vars)
NLLSModel.computeresidual(r::Counted, args...) = (Threads.atomic_add!(TALL_EVALS, 1); NLLSModel.computeresidual(r.c, args...))
Base.eltype(::Counted) = Float64

function countedchain()
    base = posechain(8, 12)
    model = Model(eltype(base.variables), Union{Counted{ChainPrior}, Counted{ChainOdometry}, Counted{ChainSighting}})
    foreach(v -> addvariable!(model, v), base.variables)
    foreach(costvec -> foreach(c -> addcost!(model, Counted(c)), costvec), values(base.costs))
    return model
end
