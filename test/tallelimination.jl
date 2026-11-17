using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization, NLPModels
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian, coupling_block, eliminated_block, residual_block, copy_structure
import OptimizationVariableInterface

# NTP2: elimination on the tall path, by QR (a SchurBlockJacobian prototype). Needs optlandmark_problem and OptLandmark from
# optimization.jl, chainproblem from frontdoor.jl, and warmallocs and AllocExt from tallalloc.jl.

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
# columns in the variable order of `plain` (a `resjac!` without elimination of the same model)
function tables_as_plain(rj, J, plain)
    lay, sch = rj.layout, rj.layout.schur
    jst = sch.structure
    ne = jst.schur.neblocks
    # The plain row of each row of each table
    rowof = [zeros(Int, BlockLinearSystems.reduced_nrows(jst) + 1000) for _ in 1:jst.ntables]
    for k in 1:lay.K, i in 1:lay.resblocksizes[k]
        rowof[sch.cost_table[k]][sch.cost_row[k] + i - 1] = lay.resoffsets[k] + i - 1
    end
    # The plain columns of each variable, by its place in the partition (retained blocks, then eliminated blocks)
    pl = plain.layout
    cols(v) = pl.varoffsets[pl.blockindices[v]] .+ (0:pl.varblocksizes[pl.blockindices[v]]-1)
    retained, eliminated = rj.unfixed_inds[1:jst.schur.nrblocks], rj.unfixed_inds[jst.schur.nrblocks+1:end]
    Jd, rd = zeros(lay.M, lay.N), zeros(lay.M)
    for t in 1:jst.ntables
        rows = filter(!iszero, rowof[t])
        for u in jst.table_ptr[t]:jst.table_ptr[t+1]-1
            Jd[rows, cols(retained[jst.entry_retained[u]])] = coupling_block(J, u)
        end
        t <= ne && (Jd[rows, cols(eliminated[t])] = eliminated_block(J, t))
        rd[rows] = residual_block(J, t)
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

@testset "tallelimination.jl" begin
    @testset "the prototype: $name" for (name, make, unfixed) in ELIM_MODELS
        model = make()
        J(; kw...) = optimvars_nls(model; unfixed, numthreads=static(1), kw...)[5]
        @test J(reduction=:eliminate, storage=:dense) isa SchurBlockJacobian{Float64, Matrix{Float64}}
        @test J(reduction=:eliminate, storage=:sparse) isa SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}
        @test J() isa SchurBlockJacobian # :auto eliminates: the model has eliminable variables and the structure fits
        @test J(reduction=:none) isa Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int}}
        # The reduced storage follows the normal path's rule: the same choice for the same model
        Hauto = optimvars(model; unfixed, numthreads=static(1))[4]
        @test (J() isa SchurBlockJacobian{Float64, Matrix{Float64}}) == (Hauto.R isa Matrix)
    end

    @testset "the same cost as the normal path: $name, $storage" for (name, make, unfixed) in ELIM_MODELS, storage in (:dense, :sparse)
        model = make()
        c0 = NLLSModel.cost(model)
        normal, _ = optimize(model, ElimOptions; unfixed, numthreads=static(1))
        tall, rt = optimize(model, ElimOptions; formulation=:tall, reduction=:eliminate, storage, unfixed, numthreads=static(1))
        plain, _ = optimize(model, ElimOptions; formulation=:tall, reduction=:none, unfixed, numthreads=static(1))
        @test NLLSModel.cost(tall) ≈ NLLSModel.cost(normal) rtol=1e-8 atol=1e-10 * c0
        @test NLLSModel.cost(tall) ≈ NLLSModel.cost(plain) rtol=1e-8 atol=1e-10 * c0
        # linsolver = :qr, a QR of the reduced Jacobian, dense or sparse (a sparse one since SecondOrderOptimization TP6)
        tallqr, _ = optimize(model, OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:qr, maxiters=200); formulation=:tall,
            reduction=:eliminate, storage, unfixed, numthreads=static(1))
        @test NLLSModel.cost(tallqr) ≈ NLLSModel.cost(normal) rtol=1e-8 atol=1e-10 * c0
        # The solution is written back through the partition (syncvars!)
        @test NLLSModel.cost(tall) ≈ rt.bestcost rtol=1e-12 atol=1e-10 * c0
        @test NLLSModel.cost(tall) < 0.1 * c0
        # Only the unfixed variables moved
        unfixed === nothing || @test all(k -> tall.variables[k] == model.variables[k], setdiff(eachindex(model.variables), unfixed))
    end

    @testset "the tables reproduce the plain Jacobian row for row: $name" for (name, make, unfixed) in ELIM_MODELS
        model = make()
        _, resp!, rjp, xp, Jp = optimvars_nls(model; unfixed, reduction=:none, storage=:dense, numthreads=static(1))
        _, rese!, rje, xe, Je = optimvars_nls(model; unfixed, reduction=:eliminate, numthreads=static(1))
        rp = zeros(size(Jp, 1))
        cp = rjp(rp, Jp, xp)
        fill!(Je.data, 7.0) # Not prezeroed: every entry is set
        ce = rje(Float64[], Je, xe) # The residual vector is not used
        @test ce == cp
        Jd, rd = tables_as_plain(rje, Je, rjp)
        @test Jd == Jp
        @test rd == rp
        # And nothing else is written: the damping rows and the entries no cost touches are zero
        @test sum(abs2, Je.data) ≈ sum(abs2, Jp) + sum(abs2, rp) rtol=1e-14
        # With prezeroed = true the zeros are taken to be in place
        Jz = copy_structure(Je)
        @test rje(Float64[], Jz, xe; prezeroed=true) == ce && Jz.data == Je.data
        # res! still writes the whole residual, in the plain order
        r2, r3 = zeros(length(rp)), zeros(length(rp))
        @test rese!(r2, xe) == resp!(r3, xp)
        @test r2 == r3
    end

    @testset "a Jacobian of another structure is refused" begin
        model = optlandmark_problem(4, 20)
        _, _, rj, x0, J = optimvars_nls(model; reduction=:eliminate, numthreads=static(1))
        J2 = optimvars_nls(model; reduction=:eliminate, numthreads=static(1))[5]
        @test_throws ArgumentError rj(Float64[], J2, x0)
        @test rj(Float64[], copy_structure(J), x0) ≈ NLLSModel.cost(model) rtol=1e-12
    end

    @testset "threads: the tall floor's rule, the same tables" begin
        nt = Threads.nthreads()
        model = posechain(40, 60)
        w = NLLSModel.workestimates(model, nothing)
        _, _, rj1, x1, J1 = optimvars_nls(model; reduction=:eliminate, numthreads=static(1))
        _, _, rjn, xn, Jn = optimvars_nls(model; reduction=:eliminate, numthreads=static(8), min_work=w.tall ÷ 2)
        @test rjn.layout.work == w.tall
        @test rjn.nthreads == NLLSModel.threadcount(w.tall, w.tall ÷ 2, 8, NLLSModel.costsplit(model.costs)) == 2
        @test optimvars_nls(model; reduction=:eliminate, numthreads=static(8))[3].nthreads ==
            NLLSModel.threadcount(w.tall, NLLSModel.MIN_WORK_PER_THREAD_TALL, 8, NLLSModel.costsplit(model.costs))
        @test rjn(Float64[], Jn, xn) ≈ rj1(Float64[], J1, x1) rtol=1e-13
        @test Jn.data == J1.data # Each cost's rows are written by one thread, the same values
    end

    @testset "zero allocation per resjac!: $name" for (name, make, unfixed) in ELIM_MODELS
        nt = Threads.nthreads()
        for storage in (:dense, :sparse)
            @test warmallocs(make(); unfixed, reduction=:eliminate, storage, numthreads=static(1), min_work=5000) == (0, 0, 0)
            @test warmallocs(make(); unfixed, reduction=:eliminate, storage, numthreads=static(nt), min_work=0) == (0, 0, 0)
        end
    end

    # A whole solve with a dense reduced system (QR of the reduced Jacobian, or Cholesky of its normal equations), serial and
    # threaded. A sparse reduced system allocates inside CHOLMOD (rules.md 21's exception), so it is not checked here.
    #
    # BlockLinearSystems' kernels thread whenever Julia has more than one thread (whatever `numthreads` says), and on Julia
    # 1.10 each threaded call may allocate (rules.md 21's exception, up to 2 048 bytes per thread per call). A whole solve
    # makes some hundred such calls, a number its result does not report, so that bound cannot be applied to a solve.
    # Fully serial (NLLSModel on one thread, BlockLinearSystems' threading off through its `SCHUR_THREAD_MIN_WORK`), a solve
    # allocates nothing on every version. With threads it allocates nothing from Julia 1.11 on; on 1.10 its threaded calls
    # are bounded per call by BlockLinearSystems' own tests, and the solve is not checked here.
    @testset "a whole solve: zero allocation, $name, $linsolver" for (name, make) in (
            ("bundle adjustment", () -> (Random.seed!(6); optlandmark_problem(6, 40))), ("pose chain with landmarks", () -> posechain(40, 60))),
            linsolver in (:auto, :cholesky)
        minwork = BlockLinearSystems.SCHUR_THREAD_MIN_WORK[]
        try
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = typemax(Int)
            @test elim_warmbytes(make; numthreads=static(1), linsolver) == 0
        finally
            BlockLinearSystems.SCHUR_THREAD_MIN_WORK[] = minwork
        end
        if VERSION >= v"1.11" || Threads.nthreads() == 1
            @test elim_warmbytes(make; numthreads=static(1), linsolver) == 0
            @test elim_warmbytes(make; numthreads=static(Threads.nthreads()), linsolver) == 0
        end
    end

    @testset "reduction = :auto and :eliminate on the tall path" begin
        ba = optlandmark_problem(4, 20)
        plain, _ = chainproblem(5, 2, 9) # No eliminable variables
        @test optimvars_nls(ba; numthreads=static(1))[5] isa SchurBlockJacobian
        @test optimvars_nls(plain; numthreads=static(1))[5] isa Matrix{Float64}
        @test_throws ArgumentError optimvars_nls(plain; numthreads=static(1), reduction=:eliminate)
        # An eliminable variable in no cost: no rows to eliminate, so :auto does not eliminate and :eliminate throws
        lonely = optlandmark_problem(4, 20)
        addvariable!(lonely, OptLandmark{Float64}(SVector(0.0, 0.0, 10.0)))
        @test (@test_logs (:warn,) match_mode=:any optimvars_nls(lonely; numthreads=static(1))[5]) isa Matrix{Float64}
        err = try optimvars_nls(lonely; numthreads=static(1), reduction=:eliminate); nothing catch e e end
        @test err isa ArgumentError && occursin("in none", err.msg)
        # Front door: :eliminate is not rejected
        @test optimize!(deepcopy(ba), ElimOptions; formulation=:tall, reduction=:eliminate, numthreads=static(1)) isa OptimizationResult
        # A single unfixed variable is never partitioned, as on the normal path
        @test optimvars_nls(ba; unfixed=1, numthreads=static(1), reduction=:eliminate)[5] isa Matrix{Float64}
        # A dense reduced system above the cap: an ArgumentError naming it (4 cameras of 6: 24×24, 4608 bytes)
        err = try optimvars_nls(ba; numthreads=static(1), reduction=:eliminate, storage=:dense, max_dense_bytes=4607); nothing catch e e end
        @test err isa ArgumentError && occursin("24×24 reduced system", err.msg)
        @test optimvars_nls(ba; numthreads=static(1), reduction=:eliminate, max_dense_bytes=4607)[5] isa SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}
    end

    @testset "the wrappers that hand a plain Jacobian on never eliminate" begin
        ba = flatelimmodel()
        @test optimvars_nls(ba; numthreads=static(1))[5] isa SchurBlockJacobian # Elimination is available on the tall path
        for (name, entry) in (("flat_optimvars_nls", flat_optimvars_nls), ("nls_callbacks", NLLSModel.nls_callbacks), ("NLLSNLSModel", NLLSNLSModel))
            err = try entry(ba; numthreads=static(1), reduction=:eliminate); nothing catch e e end
            @test err isa ArgumentError && occursin(name, err.msg) && occursin("plain Jacobian", err.msg)
        end
        @test flat_optimvars_nls(ba; numthreads=static(1)).J_proto isa Matrix{Float64} # :auto does not eliminate here
        @test NLLSModel.nls_callbacks(ba; numthreads=static(1)).J0 isa Matrix{Float64}
        @test NLLSNLSModel(ba; numthreads=static(1)).J_buf isa Matrix{Float64}
    end
end
