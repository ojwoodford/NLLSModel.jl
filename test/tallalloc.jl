using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization
import OptimizationVariableInterface: update!, copyvars

# Z4: the tall evaluators allocate nothing per call on one thread (warm). Needs tallproblems.jl and Tgt from frontdoor.jl.
# Variables of static size use autodiff; Vector variables use hand-written Jacobians (TlHand), since autodiff of a Vector variable
# allocates in `update`, in either formulation.

const AllocExt = Base.get_extension(NLLSModel, :SecondOrderOptimizationExt)

function staticrobust(kernel; n=4)
    rng = MersenneTwister(5)
    m = Model(SVector{2, Float64}, TlRob{Tgt{SVector{2, Float64}}, typeof(kernel)})
    foreach(k -> addvariable!(m, randn(rng, SVector{2, Float64})), 1:n)
    foreach(k -> addcost!(m, TlRob(Tgt(randn(rng, SVector{2, Float64}), k), kernel)), 1:n)
    return m
end
staticplain(; n=4) = (rng = MersenneTwister(5); m = Model(SVector{2, Float64}, Tgt{SVector{2, Float64}});
    foreach(k -> addvariable!(m, randn(rng, SVector{2, Float64})), 1:n);
    foreach(k -> addcost!(m, Tgt(randn(rng, SVector{2, Float64}), k)), 1:n); m)

# Warm, then the allocations of one call of each evaluator
function warmallocs(model; unfixed=nothing, numthreads=static(1), min_work_per_thread=5000)
    f, res!, resjac!, x0, J = optimvars_nls(model; unfixed, numthreads, min_work_per_thread, matrix_type=:dense)
    r = zeros(size(J, 1))
    for _ in 1:3
        resjac!(r, J, x0); res!(r, x0); f(x0)
    end
    return (@allocated resjac!(r, J, x0)), (@allocated res!(r, x0)), (@allocated f(x0))
end

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

    # The threaded launch (Polyester's argument tuple) allocates a fixed amount per call, whatever the model size
    @testset "threads: a fixed allocation per call, independent of the model" begin
        nt = Threads.nthreads()
        bigmodel(n) = (rng = MersenneTwister(3); m = Model(Vector{Float64}, TlHand{typeof(Huber2oKernel(0.3))});
            addvariable!(m, randn(rng, 12)); foreach(_ -> addcost!(m, TlHand(randn(rng, 9), 12, Huber2oKernel(0.3), 1)), 1:n); m)
        small = warmallocs(bigmodel(200); numthreads=static(nt), min_work_per_thread=100)
        large = warmallocs(bigmodel(800); numthreads=static(nt), min_work_per_thread=100)
        @test small == large
        @test all(<=(1024), small)
        nt == 1 && @test small == (0, 0, 0)
    end

    # A whole solve: per iteration, the tall path allocates no more than the normal equations do on the same model
    @testset "per iteration, no more than the normal equations: $name" for (name, make) in (
            ("static", () -> staticrobust(Huber2oKernel(0.3))), ("Vector, hand-written", () -> handmodel(Huber2oKernel(0.3), NoRobust())))
        opts(n) = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=n, reldcost=0.0, absdcost=0.0, dstep=0.0, gtol=0.0, maxfails=10^6)
        function periteration(formulation)
            function run(n)
                m = make()
                if formulation === :tall
                    f, _, rj, x0, J = optimvars_nls(m; numthreads=static(1), matrix_type=:dense)
                    obj = JacobianObjective(f, AllocExt.TallResJac(rj), J)
                else
                    f, fdf!, x0, H = optimvars(m, nothing, static(1))
                    obj = HessianObjective(f, fdf!; H)
                end
                ws = OptimizationWorkspace(x0, obj, opts(n))
                SecondOrderOptimization.optimize!(x0, obj, opts(n); workspace=ws) # warm
                a = @allocated res = SecondOrderOptimization.optimize!(x0, obj, opts(n); workspace=ws)
                return a, res.niterations
            end
            a1, i1 = run(20); a2, i2 = run(60)
            @test (i1, i2) == (20, 60) # No early termination
            return (a2 - a1) / (i2 - i1)
        end
        @test periteration(:tall) <= periteration(:normal)
    end
end
