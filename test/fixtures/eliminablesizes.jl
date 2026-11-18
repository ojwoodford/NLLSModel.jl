using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems: SchurBlockHessian, SchurBlockJacobian
import OptimizationVariableInterface
import OptimizationVariableInterface: nvars

fixture("allocations")

# An eliminable variable of N dimensions
struct SizedLandmark{N, T}
    p::SVector{N, T}
end
OptimizationVariableInterface.nvars(::SizedLandmark{N}) where N = static(N)
OptimizationVariableInterface.update(v::SizedLandmark{N}, step, offset=1) where N =
    SizedLandmark(v.p + SVector{N}(ntuple(i -> step[offset+i-1], Val(N))))
OptimizationVariableInterface.iseliminable(::Type{<:SizedLandmark}) = static(true)

const SizedPose = SVector{3, Float64}

struct SizedPrior <: NLLSModel.AbstractResidual
    t::SizedPose
    i::Int
end
NLLSModel.ndeps(::SizedPrior) = static(1)
NLLSModel.nres(::SizedPrior) = static(3)
NLLSModel.varindices(r::SizedPrior) = r.i
NLLSModel.getvars(r::SizedPrior, vars::AbstractVector) = (vars[r.i]::SizedPose,)
NLLSModel.computeresidual(r::SizedPrior, a) = a - r.t
Base.eltype(::SizedPrior) = Float64

struct SizedLink <: NLLSModel.AbstractResidual
    d::SizedPose
    i::Int
    j::Int
end
NLLSModel.ndeps(::SizedLink) = static(2)
NLLSModel.nres(::SizedLink) = static(3)
NLLSModel.varindices(r::SizedLink) = SVector(r.i, r.j)
NLLSModel.getvars(r::SizedLink, vars::AbstractVector) = (vars[r.i]::SizedPose, vars[r.j]::SizedPose)
NLLSModel.computeresidual(r::SizedLink, a, b) = b - a - r.d
Base.eltype(::SizedLink) = Float64

# A retained variable (3 dimensions) seeing an eliminated one (N dimensions): N rows, nonlinear in both
struct SizedSighting{N, L} <: NLLSModel.AbstractResidual
    z::SVector{N, Float64}
    w::SMatrix{N, 2, Float64, L}
    i::Int
    j::Int
end
NLLSModel.ndeps(::SizedSighting) = static(2)
NLLSModel.nres(::SizedSighting{N}) where N = static(N)
NLLSModel.varindices(r::SizedSighting) = SVector(r.i, r.j)
NLLSModel.getvars(r::SizedSighting{N}, vars::AbstractVector) where N = (vars[r.i]::SizedPose, vars[r.j]::SizedLandmark{N, Float64})
sizedsighting(w, p, l) = l * (1 + 0.1 * sin(p[3])) - w * p[SOneTo(2)]
NLLSModel.computeresidual(r::SizedSighting, p, l) = sizedsighting(r.w, p, l.p) - r.z
Base.eltype(::SizedSighting) = Float64

# `nposes` retained variables in a chain, `nlms` eliminated ones of N dimensions, each seen by two poses
function sizedmodel(N; nposes=6, nlms=10, seed=31)
    rng = MersenneTwister(seed)
    poses = [SizedPose(0.5k, 0.2k, 0.1k) for k in 1:nposes]
    lms = [randn(rng, SVector{N, Float64}) for _ in 1:nlms]
    model = Model(Union{SizedPose, SizedLandmark{N, Float64}}, Union{SizedPrior, SizedLink, SizedSighting{N, 2N}})
    foreach(p -> addvariable!(model, p + 0.05 * randn(rng, SizedPose)), poses)
    foreach(l -> addvariable!(model, SizedLandmark(l + 0.1 * randn(rng, SVector{N, Float64}))), lms)
    addcost!(model, SizedPrior(poses[1], 1))
    for k in 1:nposes-1
        addcost!(model, SizedLink(poses[k+1] - poses[k] + 0.01 * randn(rng, SizedPose), k, k + 1))
    end
    for (j, l) in enumerate(lms), k in (mod1(j, nposes), mod1(j + 2, nposes))
        w = randn(rng, SMatrix{N, 2, Float64})
        addcost!(model, SizedSighting(sizedsighting(w, poses[k], l) + 0.01 * randn(rng, SVector{N, Float64}), w, k, nposes + j))
    end
    return model
end

# Allocations of one warm call of the normal path's evaluators
function hessianallocs(model; numthreads, kwargs...)
    f, fdf!, x0, H = optimvars(model; numthreads, kwargs...)
    g = zeros(nvars(x0))
    return warmallocs_fdf(fdf!, g, H, x0), warmallocs_f(f, x0), fdf!.linsystem isa NLLSModel.LinearSystemShared, f.nthreads > 1
end
