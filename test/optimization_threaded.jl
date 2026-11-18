using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("allocations", "optimization", "optimizeba")

# The testsets of optimization.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "optimization.jl, threaded" begin
    # Threaded Schur linearization (disjoint landmark ownership, shared C and E) matches the serial result
    @testset "threaded Schur, $storage" for storage in (:dense, :sparse)
        Random.seed!(5)
        p_serial = optlandmark_problem(30, 120)
        p_threads = deepcopy(p_serial)
        @test NLLSModel.workestimates(p_serial, nothing).normal >= 2 * NLLSModel.MIN_WORK_PER_THREAD_NORMAL
        f1, fdf1!, x1, H1 = optimvars(p_serial; numthreads=static(1), reduction=:eliminate, storage)
        f4, fdf4!, x4, H4 = optimvars(p_threads; numthreads=static(4), reduction=:eliminate, storage)
        @test fdf4!.linsystem isa NLLSModel.LinearSystemShared
        @test fdf4!.plan isa NLLSModel.SchurThreadPlan
        n = sum(BlockLinearSystems.blocksizes(H1.structure))
        g1 = zeros(n); g4 = zeros(n)
        for _ in 1:2 # Evaluate twice: buffers must be re-zeroed
            @test fdf1!(g1, H1, x1) ≈ fdf4!(g4, H4, x4)
            @test g1 ≈ g4
            dx = randn(n)
            @test dot(dx, H1, dx) ≈ dot(dx, H4, dx)
        end
        @test f1(x1) ≈ f4(x4)
        # Floors for exactly 2 of 8 threads: 2 buffers, and the same results as one thread
        w = NLLSModel.workestimates(p_serial, nothing)
        f2, fdf2!, x2, H2 = optimvars(deepcopy(p_serial); numthreads=static(8), reduction=:eliminate, storage, min_work=w.normal ÷ 2, min_work_cost=w.cost ÷ 2)
        @test length(fdf2!.linsystem.ls) == 2 && f2.nthreads == 2
        # At most MAX_THREADS_NORMAL buffers, however many threads
        fdf8! = optimvars(deepcopy(p_serial); numthreads=static(8), reduction=:eliminate, storage, min_work=0)[2]
        @test length(fdf8!.linsystem.ls) == NLLSModel.MAX_THREADS_NORMAL
        g2 = zeros(n)
        @test fdf1!(g1, H1, x1) ≈ fdf2!(g2, H2, x2) rtol=1e-13
        @test g1 ≈ g2 rtol=1e-12
        @test f1(x1) ≈ f2(x2) rtol=1e-13
        # Threaded, a warm call allocates nothing: the linearization, and the cost (threaded too, above its floor). The
        # linearization takes the threaded path (four buffers) even on one Julia thread, so rules.md 21's 1.10 exception
        # applies to it.
        @test f4.nthreads > 1
        @test threaded_allocs_ok(warmallocs_fdf(fdf4!, g4, H4, x4), length(fdf4!.linsystem.ls) > 1)
        @test warmallocs_f(f4, x4) == 0
    end
end
