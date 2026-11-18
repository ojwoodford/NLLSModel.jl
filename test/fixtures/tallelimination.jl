using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization, NLPModels
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian, block, blockentry, residual_block, copy_structure
import OptimizationVariableInterface

fixture("optimization", "tallalloc")

# A pose chain with landmarks. Poses (x, y, θ) are retained, linked by odometry and anchored by a prior on the first; landmarks
# (x, y) are eliminable, each seen by two or three poses. The odometry and the prior involve no eliminated variable, so their
# rows go in free tables: one per set of retained variables.
struct ChainLandmark{T}
    p::SVector{2, T}
end
OptimizationVariableInterface.nvars(::ChainLandmark) = static(2)
OptimizationVariableInterface.update(v::ChainLandmark, step, offset=1) = ChainLandmark(v.p + SVector(step[offset], step[offset+1]))
OptimizationVariableInterface.iseliminable(::Type{<:ChainLandmark}) = static(true)

const Pose2 = SVector{3, Float64}
rot2(θ) = SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
relpose(a, b) = vcat(rot2(a[3])' * (b[SOneTo(2)] - a[SOneTo(2)]), b[3] - a[3])
sighting(p, l) = rot2(p[3])' * (l - p[SOneTo(2)])

struct ChainPrior <: NLLSModel.AbstractResidual
    t::Pose2
    i::Int
end
NLLSModel.ndeps(::ChainPrior) = static(1)
NLLSModel.nres(::ChainPrior) = static(3)
NLLSModel.varindices(r::ChainPrior) = r.i
NLLSModel.getvars(r::ChainPrior, vars::AbstractVector) = (vars[r.i]::Pose2,)
NLLSModel.computeresidual(r::ChainPrior, a) = a - r.t
Base.eltype(::ChainPrior) = Float64

struct ChainOdometry <: NLLSModel.AbstractResidual
    d::Pose2
    i::Int
    j::Int
end
NLLSModel.ndeps(::ChainOdometry) = static(2)
NLLSModel.nres(::ChainOdometry) = static(3)
NLLSModel.varindices(r::ChainOdometry) = SVector(r.i, r.j)
NLLSModel.getvars(r::ChainOdometry, vars::AbstractVector) = (vars[r.i]::Pose2, vars[r.j]::Pose2)
NLLSModel.computeresidual(r::ChainOdometry, a, b) = relpose(a, b) - r.d
Base.eltype(::ChainOdometry) = Float64

struct ChainSighting <: NLLSModel.AbstractResidual
    z::SVector{2, Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::ChainSighting) = static(2)
NLLSModel.nres(::ChainSighting) = static(2)
NLLSModel.varindices(r::ChainSighting) = SVector(r.i, r.j)
NLLSModel.getvars(r::ChainSighting, vars::AbstractVector) = (vars[r.i]::Pose2, vars[r.j]::ChainLandmark{Float64})
NLLSModel.computeresidual(r::ChainSighting, p, l) = sighting(p, l.p) - r.z
Base.eltype(::ChainSighting) = Float64

function posechain(nposes, nlms; seed=21)
    rng = MersenneTwister(seed)
    poses = [Pose2(0.8k, 0.3sin(k), 0.15k) for k in 1:nposes]
    lms = [SVector(0.8 * nposes * rand(rng), 2 * randn(rng)) for _ in 1:nlms]
    model = Model(Union{Pose2, ChainLandmark{Float64}}, Union{ChainPrior, ChainOdometry, ChainSighting})
    foreach(p -> addvariable!(model, p + 0.05 * randn(rng, Pose2)), poses)
    foreach(l -> addvariable!(model, ChainLandmark(l + 0.1 * randn(rng, SVector{2, Float64}))), lms)
    addcost!(model, ChainPrior(poses[1], 1))
    for k in 1:nposes-1
        addcost!(model, ChainOdometry(relpose(poses[k], poses[k+1]) + 0.01 * randn(rng, Pose2), k, k + 1))
    end
    for (j, l) in enumerate(lms)
        for k in sortperm([abs(p[1] - l[1]) for p in poses])[1:2+j%2]
            addcost!(model, ChainSighting(sighting(poses[k], l) + 0.01 * randn(rng, SVector{2, Float64}), k, nposes + j))
        end
    end
    return model
end

# Bundle adjustment with one more landmark, seen by one camera only: two rows for three dimensions (an under-determined block)
function ba_underdetermined()
    model = optlandmark_problem(4, 20)
    l = addvariable!(model, OptLandmark{Float64}(SVector(0.1, -0.2, 10.0)))
    z = NLLSModel.generatemeasurement(model.variables[1]::SVector{6, Float64}, model.variables[l]) .+ 0.01
    addcost!(model, SimpleError2{SVector{6, Float64}, OptLandmark{Float64}}(z, 1, l))
    return model
end

# An eliminable variable that packs into a flat vector (an AbstractVector built from its values), for the flat wrappers
struct FlatLandmark{T} <: AbstractVector{T}
    p::SVector{2, T}
    FlatLandmark{T}(p::SVector{2, T}) where T = new{T}(p)
end
FlatLandmark(p::SVector{2, T}) where T = FlatLandmark{T}(p)
FlatLandmark{T}(v::AbstractVector) where T = FlatLandmark{T}(SVector{2, T}(v[1], v[2]))
Base.size(::FlatLandmark) = (2,)
Base.getindex(l::FlatLandmark, i::Int) = l.p[i]
OptimizationVariableInterface.nvars(::FlatLandmark) = static(2)
OptimizationVariableInterface.update(v::FlatLandmark, step, offset=1) = FlatLandmark(v.p + SVector(step[offset], step[offset+1]))
OptimizationVariableInterface.iseliminable(::Type{<:FlatLandmark}) = static(true)

struct FlatSighting <: NLLSModel.AbstractResidual
    z::SVector{2, Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::FlatSighting) = static(2)
NLLSModel.nres(::FlatSighting) = static(2)
NLLSModel.varindices(r::FlatSighting) = SVector(r.i, r.j)
NLLSModel.getvars(r::FlatSighting, vars::AbstractVector) = (vars[r.i]::Pose2, vars[r.j]::FlatLandmark{Float64})
NLLSModel.computeresidual(r::FlatSighting, p, l) = l.p - p[SOneTo(2)] - r.z
Base.eltype(::FlatSighting) = Float64

function flatelimmodel()
    rng = MersenneTwister(4)
    model = Model(Union{Pose2, FlatLandmark{Float64}}, FlatSighting)
    foreach(_ -> addvariable!(model, randn(rng, Pose2)), 1:3)
    foreach(_ -> addvariable!(model, FlatLandmark(randn(rng, SVector{2, Float64}))), 1:4)
    for c in 1:3, l in 1:4
        addcost!(model, FlatSighting(randn(rng, SVector{2, Float64}), c, 3 + l))
    end
    return model
end

const ElimOptions = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200)
const ELIM_MODELS = (
    ("bundle adjustment", () -> optlandmark_problem(4, 20), nothing),
    ("bundle adjustment, an under-determined block", ba_underdetermined, nothing),
    ("pose chain with landmarks", () -> posechain(8, 12), nothing),
    # Pose 1 and landmark 1 fixed: the prior's rows involve no unfixed variable (a free table with no columns)
    ("pose chain, two variables fixed", () -> posechain(8, 12), [2:8; 10:20]))

# The tables of a SchurBlockJacobian as one plain Jacobian and residual, in the row order of the plain tall path, and the
# columns in the variable order of `plain` (a `resjac!` without elimination of the same model): each cost is a row block,
# read through blockentry, block and residual_block
function tables_as_plain(rj, J, plain)
    lay, pl = rj.layout, plain.layout
    jst = J.structure
    layoutblock = zeros(Int, length(rj.problem.variables))   # Layout block of each unfixed variable
    for (p, v) in enumerate(rj.unfixed_inds)
        layoutblock[v] = p
    end
    sizes = BlockLinearSystems.blocksizes(jst)
    cols(v) = pl.varoffsets[pl.blockindices[v]] .+ (0:pl.varblocksizes[pl.blockindices[v]]-1)
    Jd, rd = zeros(lay.M, lay.N), zeros(lay.M)
    NLLSModel.eachcost(rj.problem.costs) do c, k
        m = lay.resblocksizes[k]
        rows = lay.resoffsets[k] .+ (0:m-1)
        vinds = NLLSModel.varindices(c)
        for v in (vinds isa Integer ? (vinds,) : vinds)
            j = layoutblock[v]
            j == 0 && continue
            Jd[rows, cols(v)] = block(J, k, blockentry(jst, k, j), m, sizes[j])
        end
        rd[rows] = residual_block(J, k)
    end
    return Jd, rd
end

# Bytes allocated by a whole solve through the optimizer: the third of three identical runs (the same start, the same
# workspace), so that every branch the run takes is compiled; measured through a function barrier, as in tallalloc.jl
measure_run(x0, obj, opts, ws) = (a = @allocated(res = SecondOrderOptimization.optimize!(x0, obj, opts; workspace=ws)); (a, res.niterations))
function elim_warmbytes(make; numthreads, storage=:dense, linsolver=:auto, n=30)
    opts = OptimizationOptions(iterator=:levenbergmarquardt, linsolver=linsolver, maxiters=n, reldcost=0.0, absdcost=0.0, dstep=0.0, gtol=0.0, maxstall=10^6,
        maxtime=10^6) # No time limit: on a loaded machine a run must not stop early
    f, _, rj, x0, J = optimvars_nls(make(); numthreads, reduction=:eliminate, storage, min_work=0)
    obj = JacobianObjective(f, AllocExt.TallResJac(rj), J)
    ws = OptimizationWorkspace(x0, obj, opts)
    start = copy(x0)
    runs = map(1:3) do _
        copyto!(x0, start)
        measure_run(x0, obj, opts, ws)
    end
    @test all(r -> r[2] == n, runs) # No early termination
    return runs[3][1]
end
