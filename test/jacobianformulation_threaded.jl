using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars

include("fixtures/loader.jl")
fixture("frontdoor", "jacobianformulation", "jacobianproblems")

# The testsets of jacobianformulation.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "jacobianformulation.jl, threaded" begin
    @testset "keywords: used by :jacobian" begin
        model = curvefit()
        # numthreads
        a, _ = optimize(model, TallOptions; formulation=:jacobian, numthreads=static(1), min_work=1)
        b, _ = optimize(model, TallOptions; formulation=:jacobian, numthreads=static(4), min_work=1)
        @test NLLSModel.cost(a) ≈ NLLSModel.cost(b) rtol=1e-10
        # callback
        calls = Ref(0)
        counter(cost, unused...) = (calls[] += 1; (cost, 0))
        optimize(model, TallOptions; formulation=:jacobian, numthreads=static(1), callback=counter)
        @test calls[] > 0
        # workspace: built for the same objective and reused
        m = deepcopy(model)
        f, _, resjac!, x0, J = tlnls(m; numthreads=static(1), storage=:dense)
        obj = JacobianObjective(f, TallExt.TallResJac(resjac!), J)
        ws = OptimizationWorkspace(x0, obj, TallOptions)
        r1 = optimize!(deepcopy(model), TallOptions; formulation=:jacobian, numthreads=static(1), workspace=ws)
        r2 = optimize!(deepcopy(model), TallOptions; formulation=:jacobian, numthreads=static(1), workspace=ws)
        @test r1.bestcost == r2.bestcost
        @test r1.bestcost ≈ optimize(model, TallOptions; formulation=:jacobian, numthreads=static(1))[2].bestcost rtol=1e-10
        # min_work: the same answer however it is split
        c, _ = optimize(model, TallOptions; formulation=:jacobian, numthreads=static(1), min_work=10^9)
        @test NLLSModel.cost(c) ≈ NLLSModel.cost(a) rtol=1e-10
    end

    @testset "keywords: min_work_cost, the floor of the cost function (A2)" begin
        @testset "$(nvars) variable(s)" for nvars in (1, 2)
            model = spymodel(400, nvars)
            x = nvars == 1 ? 0.3 : [0.3, 0.0]
            f0 = optimvars_nls(model; numthreads=static(4), min_work_cost=0)[1]
            fbig = optimvars_nls(model; numthreads=static(4), min_work_cost=10^9)[1]
            @test f0 isa (nvars == 1 ? NLLSModel.SingleVarCostFunc : NLLSModel.MultiVarCostFunc)
            @test f0.work >= f0.min_work == 0 # Threaded
            @test fbig.work < fbig.min_work   # Not
            @test optimvars_nls(model)[1].min_work == NLLSModel.MIN_WORK_PER_THREAD_COST # Default unchanged
            @test f0(x) ≈ fbig(x) rtol=1e-13
            flat0 = flat_optimvars_nls(model; numthreads=static(4), min_work_cost=0)
            flatbig = flat_optimvars_nls(model; numthreads=static(4), min_work_cost=10^9)
            @test flat0.f(flat0.x0) ≈ flatbig.f(flatbig.x0) rtol=1e-13
            if Threads.nthreads() > 1
                @test offthread(() -> f0(x)) && !offthread(() -> fbig(x))
                @test offthread(() -> flat0.f(flat0.x0)) && !offthread(() -> flatbig.f(flatbig.x0))
            end
        end
        # nls_callbacks (internal) passes it on too
        cb0 = NLLSModel.nls_callbacks(spymodel(400); numthreads=static(4), min_work_cost=0)
        cbbig = NLLSModel.nls_callbacks(spymodel(400); numthreads=static(4), min_work_cost=10^9)
        @test cb0.f([0.3]) ≈ cbbig.f([0.3]) rtol=1e-13
        Threads.nthreads() > 1 && @test offthread(() -> cb0.f([0.3])) && !offthread(() -> cbbig.f([0.3]))
        # optimize!: the residual and Jacobian fill held serial, so only the cost function can thread
        model = spymodel(400)
        solve(min_work_cost) = optimize(model, TallOptions; formulation=:jacobian, numthreads=static(4), min_work=10^9, min_work_cost)
        @test NLLSModel.cost(solve(0)[1]) ≈ NLLSModel.cost(solve(10^9)[1]) rtol=1e-10
        if Threads.nthreads() > 1
            @test offthread(() -> solve(0)) && !offthread(() -> solve(10^9))
        end
    end

    @testset "threads: the same result however many" begin
        model = posemodel()
        a, ra = optimize(model, TallOptions; formulation=:jacobian, numthreads=static(1), min_work=1)
        b, rb = optimize(model, TallOptions; formulation=:jacobian, numthreads=static(Threads.nthreads()), min_work=1)
        @test NLLSModel.cost(a) ≈ NLLSModel.cost(b) rtol=1e-10
        @test samevars(a, b; atol=1e-8)
    end
end
