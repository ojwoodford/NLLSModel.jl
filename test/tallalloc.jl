using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import OptimizationVariableInterface: update, update!, copyvars

include("fixtures/loader.jl")
fixture("allocations", "frontdoor", "tallalloc", "tallproblems")

# Z4: the tall evaluators allocate nothing per call on one thread (warm).
# Variables of static size use autodiff; Vector variables use hand-written Jacobians (TlHand), since autodiff of a Vector variable
# allocates in `update`, in either formulation.

@testset "tallalloc.jl" begin
    @testset "one thread, a warm call allocates nothing: static variables, $name" for (name, make) in (
            ("no kernel", staticplain), ("Huber", () -> staticrobust(HuberKernel(0.3))), ("Huber2o", () -> staticrobust(Huber2oKernel(0.3))),
            ("Geman-McClure 2o", () -> staticrobust(GemanMcclure2oKernel(0.3))))
        @test warmallocs(make()) == (0, 0, 0)
    end

    @testset "one thread, a warm call allocates nothing: Vector variables, $name" for (name, k1, k2) in (
            ("no kernel", NoRobust(), NoRobust()), ("Huber", HuberKernel(0.3), NoRobust()), ("Huber2o", Huber2oKernel(0.3), HuberKernel(0.3)),
            ("Geman-McClure 2o", GemanMcclure2oKernel(0.3), Huber2oKernel(0.3)), ("a scaled linear kernel", Scaled(NoRobust(), 2.0), NoRobust()))
        model = handmodel(k1, k2)
        for unfixed in (nothing, [2], 1, Bool[1, 0])
            a = warmallocs(model; unfixed)
            # (`f`, the cost function shared with the normal equations, allocates 16 B for a Scaled kernel; not an evaluator)
            @test a[1:2] == (0, 0)
            name == "a scaled linear kernel" || @test a[3] == 0
        end
    end

    @testset "one thread: a subset unfixed, static variables" begin
        @test warmallocs(staticrobust(Huber2oKernel(0.3); n=6); unfixed=[1, 3, 5]) == (0, 0, 0)
        @test warmallocs(staticplain(n=6); unfixed=2) == (0, 0, 0)
    end

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

    # NTP3: the optimizer's static system hands resjac! an MMatrix and an MVector; a warm call allocates nothing (through a
    # function barrier), and neither does a warm solve through the static system
    @testset "the static system's MMatrix: a warm call allocates nothing, $name" for (name, kernel) in (("no kernel", NoRobust()),
            ("Huber2o", Huber2oKernel(0.3)))
        rng = MersenneTwister(8)
        model = Model(SVector{3, Float64}, TlRob{Tgt{SVector{3, Float64}}, typeof(kernel)})
        addvariable!(model, randn(rng, SVector{3, Float64}))
        foreach(_ -> addcost!(model, TlRob(Tgt(randn(rng, SVector{3, Float64}), 1), kernel)), 1:4) # 12 × 3: under the cap
        f, _, rj, x0, J = optimvars_nls(model; numthreads=static(1))
        @test staticallocs(rj, MVector{12}(zeros(12)), MMatrix{12, 3}(zeros(12, 3)), x0) == (0, 0)
        @test staticsolve_allocs(f, rj, x0, J) == 0
    end

    # A whole solve allocates exactly what the model's variables must, in either formulation. The optimizer makes each trial
    # point with `update`, once per variable per linear solve; for a `Vector` variable that builds a new vector
    # (OptimizationVariableInterface's `update`), for a static one nothing. Nothing else allocates, so the bytes are
    # `result.linearsolves` times one `update` of every variable, whatever steps the iterator takes.
    @testset "a whole solve allocates only the variables' updates: $name, $formulation" for (name, make) in (
            ("static", () -> staticrobust(Huber2oKernel(0.3))), ("Vector, hand-written", () -> handmodel(Huber2oKernel(0.3), NoRobust()))),
            formulation in (:tall, :normal)
        opts = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=60, reldcost=0.0, absdcost=0.0, dstep=0.0, gtol=0.0, maxstall=10^6)
        m = make()
        if formulation === :tall
            f, _, rj, x0, J = optimvars_nls(m; numthreads=static(1), storage=:dense)
            obj = JacobianObjective(f, AllocExt.TallResJac(rj), J)
        else
            f, fdf!, x0, H = optimvars(m; numthreads=static(1))
            obj = HessianObjective(f, fdf!; H)
        end
        a, res = solveallocs(x0, obj, opts, OptimizationWorkspace(x0, obj, opts))
        @test res.niterations == 60 # No early termination
        step = zeros(nvars(x0))
        perupdate = sum(v -> allocs_warm(update, v, step, 1), x0)
        @test (perupdate == 0) == (name == "static")
        @test a == res.linearsolves * perupdate
    end
end
