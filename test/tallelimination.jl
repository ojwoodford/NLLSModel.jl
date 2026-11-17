using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization, NLPModels
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian, coupling_block, eliminated_block, residual_block, copy_structure
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("allocations", "frontdoor", "tallelimination")

# NTP2: elimination on the tall path, by QR (a SchurBlockJacobian prototype).

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
