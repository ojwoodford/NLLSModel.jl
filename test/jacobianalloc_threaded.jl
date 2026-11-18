using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import OptimizationVariableInterface: update, update!, copyvars

include("fixtures/loader.jl")
fixture("allocations", "frontdoor", "jacobianalloc", "jacobianproblems")

# The testsets of jacobianalloc.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "jacobianalloc.jl, threaded" begin
    # Threaded, a warm call allocates nothing either (the threads' arguments are passed by `threadargs`)
    @testset "threads: a warm call allocates nothing" begin
        nt = Threads.nthreads()
        bigmodel(n) = (rng = MersenneTwister(3); m = Model(Vector{Float64}, TlHand{typeof(Huber2oKernel(0.3))});
            addvariable!(m, randn(rng, 12)); foreach(_ -> addcost!(m, TlHand(randn(rng, 9), 12, Huber2oKernel(0.3), 1)), 1:n); m)
        @test warmallocs(bigmodel(200); numthreads=static(nt), min_work=100) == (0, 0, 0)
        @test warmallocs(bigmodel(800); numthreads=static(nt), min_work=100) == (0, 0, 0)
    end

    # NTP1: a sparse Jacobian, serial and threaded
    @testset "a sparse Jacobian: a warm call allocates nothing, $threads" for threads in ("one thread", "threads")
        nt = threads == "one thread" ? 1 : Threads.nthreads()
        kw = (numthreads=static(nt), min_work=(nt == 1 ? 5000 : 0), storage=:sparse)
        _, _, rj, _, Jsp = optimvars_nls(staticplain(n=40); kw...)
        @test Jsp isa SparseMatrixCSC && rj.nthreads == nt # the fill is threaded when there are threads
        @test warmallocs(staticrobust(Huber2oKernel(0.3); n=40); kw...) == (0, 0, 0)
        @test warmallocs(staticplain(n=40); unfixed=[1, 3, 5], kw...) == (0, 0, 0)
        @test warmallocs(handmodel(Huber2oKernel(0.3), HuberKernel(0.3)); kw...)[1:2] == (0, 0)
    end
end
