using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization, NLPModels
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian, block, blockentry, residual_block, copy_structure
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("allocations", "frontdoor", "tallelimination")

# The testsets of tallelimination.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "tallelimination.jl, threaded" begin
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
    # `numthreads` reaches BlockLinearSystems' kernels through the structure's `nchunks`: with `numthreads = 1` they run
    # serially too, and a solve allocates nothing on every version. With threads, on Julia 1.10 each threaded kernel call
    # may allocate (rules.md 21's exception, up to 2 048 bytes per thread per call); a whole solve makes some hundred
    # such calls, a number its result does not report, so that bound cannot be applied to a solve: from Julia 1.11 on it
    # allocates nothing, and on 1.10 its threaded calls are bounded per call by BlockLinearSystems' own tests.
    @testset "a whole solve: zero allocation, $name, $linsolver" for (name, make) in (
            ("bundle adjustment", () -> (Random.seed!(6); optlandmark_problem(6, 40))), ("pose chain with landmarks", () -> posechain(40, 60))),
            linsolver in (:auto, :cholesky)
        @test elim_warmbytes(make; numthreads=static(1), linsolver) == 0
        if VERSION >= v"1.11" || Threads.nthreads() == 1
            @test elim_warmbytes(make; numthreads=static(Threads.nthreads()), linsolver) == 0
        end
    end
end
