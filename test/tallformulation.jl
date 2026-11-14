using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization
import OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars

# optimize!(model, options; formulation = :tall): the residual-and-Jacobian path. Needs Tgt, Lnk, singleproblem, mixedproblem
# and chainproblem from frontdoor.jl, and tallproblems.jl.

const TallExt = Base.get_extension(NLLSModel, :SecondOrderOptimizationExt)
const TallOptions = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200)

# Compare the tall solve with the normal equations: same final cost (the gradient at an accurate minimum is rounding noise, so it
# is not compared), and the variables to `atol`. `cost0` is the cost at the start, to give the tolerance for zero-residual problems.
function sameoptimum(model, options=TallOptions; atol=1e-5, kwargs...)
    c0 = NLLSModel.cost(model)
    normal, rn = optimize(model, options; numthreads=static(1), kwargs...)
    tall, rt = optimize(model, options; formulation=:tall, numthreads=static(1), kwargs...)
    cn, ct = NLLSModel.cost(normal), NLLSModel.cost(tall)
    ok = isapprox(ct, cn; rtol=1e-8, atol=1e-10 * c0) && isapprox(ct, rt.bestcost; rtol=1e-12) && samevars(tall, normal; atol) && rt.bestcost < c0
    return ok, ct, cn
end

# --- Tests ------------------------------------------------------------------------------------------------------------

@testset "tallformulation.jl" begin
    @test TallExt isa Module

    @testset "formulation is a keyword of optimize! only" begin
        @test !(:formulation in fieldnames(typeof(TallOptions)))
        model = curvefit()
        @test_throws ArgumentError optimize!(model, TallOptions; formulation=:sparse)
        @test_throws ArgumentError optimize!(model, TallOptions; formulation=:tall, reduction=:bogus)
        # Not inferred from the linear solver: asking for the QR solver on the normal equations is the optimizer's business
        r = optimize!(deepcopy(model), OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:qr, maxiters=200); formulation=:normal, numthreads=static(1))
        @test r isa OptimizationResult
        # :normal is the default
        a, ra = optimize(model, TallOptions; numthreads=static(1))
        b, rb = optimize(model, TallOptions; formulation=:normal, numthreads=static(1))
        @test a.variables == b.variables
        @test ra.bestcost == rb.bestcost
    end

    @testset "same optimum as :normal: $name" for (name, make) in (
            ("curve fit, one Vector", () -> curvefit()),
            ("several variables of different sizes", () -> first(mixedproblem())),
            ("chain of Vectors", () -> first(chainproblem(6, 3, 5))),
            ("rotation", () -> rotproblem()),
            ("rotation, Vector, SVector, Vector", () -> posemodel()),
            ("Huber on every cost", () -> curvefit(huber=true, outliers=4)),
            ("Huber on every cost, mixed variables", () -> posemodel(huber=true)))
        ok, ct, cn = sameoptimum(make())
        @test ok
    end

    @testset "same optimum as :normal: a single static variable of size 3 (through optimize)" begin
        model, t = singleproblem(SVector{3, Float64}, 3, 7)
        ok, _, _ = sameoptimum(model)
        @test ok
        solved, result = optimize(model, TallOptions; formulation=:tall)
        @test solved.variables[1] ≈ t atol=1e-6
        @test NLLSModel.cost(solved) ≈ result.bestcost atol=1e-12
        # optimize! of a single static variable too
        m2, t2 = singleproblem(SVector{3, Float64}, 3, 8)
        r = optimize!(m2, TallOptions; formulation=:tall)
        @test m2.variables[1] ≈ t2 atol=1e-6
        @test NLLSModel.cost(m2) ≈ r.bestcost atol=1e-12
    end

    @testset "same optimum, robust Huber with outliers is not the least-squares optimum" begin
        # Guard against the Huber tests above passing because the kernel was silently ignored
        ls = optimize(curvefit(outliers=4), TallOptions; formulation=:tall, numthreads=static(1))[1]
        hub = optimize(curvefit(huber=true, outliers=4), TallOptions; formulation=:tall, numthreads=static(1))[1]
        @test norm(ls.variables[1] - hub.variables[1]) > 1e-3
    end

    @testset "write-back (T2): the model holds the best point, not the last evaluated" begin
        # The evaluators copy every point they evaluate into the model. A Levenberg-Marquardt run in practice ends on an accepted
        # trial, so a rejected last evaluation is simulated: a callback that, like an evaluator at a rejected trial, leaves a
        # worse point in the model after every iteration. Whatever it leaves, optimize! must put the best point in. (With every variable
        # of several unfixed the evaluators write nothing into the model, x0 being model.variables itself: nothing to simulate.)
        junk(v::Vector{Float64}) = v .+ 0.5
        junk(v::SVector) = v .+ 0.5
        junk(v::TlRot) = update(v, [0.3, -0.2, 0.1])
        for (name, make, unfixed) in (("one variable", () -> curvefit(huber=true, outliers=3), nothing), ("a list", () -> posemodel(), [1, 3, 4]), ("a rotation", () -> posemodel(), 2))
            for maxiters in (1, 2, 3, 200)
                model = make()
                lastjunk = Ref{Any}(nothing)
                function leavejunk(cost, x_trial, unused...)
                    k = unfixed isa Vector ? unfixed[1] : something(unfixed, 1)
                    model.variables[k] = junk(model.variables[k])
                    lastjunk[] = model.variables[k]
                    return cost, 0
                end
                result = optimize!(model, OptimizationOptions(iterator=:levenbergmarquardt, maxiters=maxiters); formulation=:tall, numthreads=static(1), callback=leavejunk, unfixed)
                @test NLLSModel.cost(model) ≈ result.bestcost rtol=1e-12
                k = unfixed isa Vector ? unfixed[1] : something(unfixed, 1)
                @test model.variables[k] != lastjunk[]
            end
        end
        # Without it the model would be left on the last point evaluated: run the optimizer on the evaluators directly
        m = posemodel()
        f, _, resjac!, x0, J = tlnls(m; unfixed=[1, 3], numthreads=static(1), matrix_type=:dense)
        obj = JacobianObjective(f, TallExt.PrezeroedResJac(resjac!), J)
        result = SecondOrderOptimization.optimize!(x0, obj, TallOptions)
        resjac!(zeros(size(J, 1)), J, update!(deepcopy(x0), x0, fill(0.3, 4))) # A rejected trial, the last thing evaluated
        @test !(NLLSModel.cost(m) ≈ result.bestcost)
        syncvars!(m, resjac!, x0)
        @test NLLSModel.cost(m) ≈ result.bestcost rtol=1e-12
        # Each unfixed layout: one variable, a list, and all
        for unfixed in (3, 2, [1, 4], nothing)
            m = posemodel()
            f, res!, resjac!, x0, J = tlnls(m; unfixed, numthreads=static(1), matrix_type=:dense)
            x = update!(deepcopy(x0), x0, fill(0.1, size(J, 2)))
            for ev in (res!, resjac!)
                resjac!(zeros(size(J, 1)), J, x0) # The model holds x0
                syncvars!(m, ev, x)
                f2, = tlnls(m; unfixed, numthreads=static(1), matrix_type=:dense)
                @test f2(x) ≈ f(x)
                @test NLLSModel.cost(m) ≈ f(x)
            end
        end
    end

    @testset "unfixed = $name" for (name, unfixed, free) in (
            ("nothing", nothing, 1:4),
            ("an index", 4, [4]),
            ("an index of a rotation", 2, [2]),
            ("a list", [1, 3], [1, 3]),
            ("a type", Vector{Float64}, [1, 4]),
            ("a mask", Bool[0, 1, 1, 1], [2, 3, 4]))
        model = posemodel()
        start = deepcopy(model.variables)
        result = optimize!(model, TallOptions; formulation=:tall, unfixed, numthreads=static(1))
        for k in setdiff(1:4, free)
            @test model.variables[k] == start[k] # Fixed variables are untouched, bit for bit
        end
        # The free variables are at the optimum: the same as the normal equations find
        ref = deepcopy(posemodel())
        optimize!(ref, TallOptions; unfixed, numthreads=static(1))
        @test NLLSModel.cost(model) ≈ NLLSModel.cost(ref) rtol=1e-8
        @test NLLSModel.cost(model) ≈ result.bestcost atol=1e-12
        for k in free
            @test isapprox(model.variables[k], ref.variables[k]; atol=1e-5)
        end
        @test any(k -> model.variables[k] != start[k], free)
    end

    @testset "keywords: used by :tall" begin
        model = curvefit()
        # numthreads
        a, _ = optimize(model, TallOptions; formulation=:tall, numthreads=static(1), min_work_per_thread=1)
        b, _ = optimize(model, TallOptions; formulation=:tall, numthreads=static(4), min_work_per_thread=1)
        @test NLLSModel.cost(a) ≈ NLLSModel.cost(b) rtol=1e-10
        # callback
        calls = Ref(0)
        counter(cost, unused...) = (calls[] += 1; (cost, 0))
        optimize(model, TallOptions; formulation=:tall, numthreads=static(1), callback=counter)
        @test calls[] > 0
        # workspace: built for the same objective and reused
        m = deepcopy(model)
        f, _, resjac!, x0, J = tlnls(m; numthreads=static(1), matrix_type=:dense)
        obj = JacobianObjective(f, TallExt.PrezeroedResJac(resjac!), J)
        ws = OptimizationWorkspace(x0, obj, TallOptions)
        r1 = optimize!(deepcopy(model), TallOptions; formulation=:tall, numthreads=static(1), workspace=ws)
        r2 = optimize!(deepcopy(model), TallOptions; formulation=:tall, numthreads=static(1), workspace=ws)
        @test r1.bestcost == r2.bestcost
        @test r1.bestcost ≈ optimize(model, TallOptions; formulation=:tall, numthreads=static(1))[2].bestcost rtol=1e-10
        # min_work_per_thread: the same answer however it is split
        c, _ = optimize(model, TallOptions; formulation=:tall, numthreads=static(1), min_work_per_thread=10^9)
        @test NLLSModel.cost(c) ≈ NLLSModel.cost(a) rtol=1e-10
    end

    @testset "keywords: reduction" begin
        model = curvefit()
        for reduction in (:auto, :none)
            @test optimize!(deepcopy(model), TallOptions; formulation=:tall, reduction, numthreads=static(1)) isa OptimizationResult
        end
        err = try optimize!(deepcopy(model), TallOptions; formulation=:tall, reduction=:eliminate); nothing catch e e end
        @test err isa ArgumentError
        @test occursin("eliminate", err.msg) && occursin("tall", err.msg)
        # and the normal formulation still takes all three
        @test optimize!(posemodel(), TallOptions; reduction=:none, numthreads=static(1)) isa OptimizationResult
    end

    @testset "keywords: ignored by :tall are accepted silently" begin
        model = curvefit()
        @test_logs optimize!(deepcopy(model), TallOptions; formulation=:tall, numthreads=static(1), reduced=:sparse, min_work=1, min_work_cost=1)
        @test_logs optimize!(deepcopy(model), TallOptions; formulation=:tall, numthreads=static(1), reduced=:dense)
    end

    @testset "keywords: ignored by :normal are accepted silently" begin
        model = curvefit()
        ref, rr = optimize(model, TallOptions; numthreads=static(1))
        @test_logs (out = optimize(model, TallOptions; numthreads=static(1), min_work_per_thread=3, max_dense_bytes=1))
        out, rout = optimize(model, TallOptions; numthreads=static(1), min_work_per_thread=3, max_dense_bytes=1)
        @test out.variables == ref.variables # Nothing changed
        @test rout.bestcost == rr.bestcost
    end

    @testset "kernels (R1): the kernel decides, there is no robust keyword" begin
        @test_throws MethodError optimize!(curvefit(), TallOptions; formulation=:tall, robust=:irls)
        @test_throws MethodError optimize!(curvefit(), TallOptions; robust=:none)
        # NoRobust, Huber and Huber2o costs in one model reach the same optimum as the normal equations
        model = Model(Vector{Float64}, Union{TlExp, TlRob{TlExp, typeof(HuberKernel(0.05))}, TlRob{TlExp, typeof(Huber2oKernel(0.05))}})
        base = curvefit(outliers=4)
        addvariable!(model, base.variables[1])
        for (k, c) in enumerate(costlist(base))
            addcost!(model, k % 3 == 0 ? c : k % 3 == 1 ? TlRob(c, HuberKernel(0.05)) : TlRob(c, Huber2oKernel(0.05)))
        end
        ok, ct, cn = sameoptimum(model)
        @test ok
        @test ct ≈ cn rtol=1e-8
        # it is not the least-squares optimum
        @test norm(optimize(model, TallOptions; formulation=:tall)[1].variables[1] - optimize(base, TallOptions; formulation=:tall)[1].variables[1]) > 1e-3
        # Each kernel on its own
        for h in (true, :second)
            ok, = sameoptimum(curvefit(huber=h, outliers=4))
            @test ok
        end
        # A fixed adaptive kernel works with no mode to choose
        prob = Model(Union{ContaminatedGaussian{Float64}, Float64}, TlAdapt)
        addvariable!(prob, ContaminatedGaussian(1.0, 5.0, 0.8))
        addvariable!(prob, 2.0)
        addcost!(prob, TlAdapt(SVector(1, 2)))
        out, res = optimize(prob, TallOptions; formulation=:tall, unfixed=2, numthreads=static(1))
        @test out.variables[2] ≈ 1.0 atol=1e-6
    end

    @testset "unsupported models fail clearly, naming the tall formulation" begin
        # A cost that is not a residual
        model = Model(Vector{Float64}, TlCost)
        addvariable!(model, [1.0, 2.0])
        addcost!(model, TlCost(1))
        err = try optimize!(model, TallOptions; formulation=:tall); nothing catch e e end
        @test err isa ArgumentError
        @test occursin("tall", err.msg) && occursin("AbstractResidual", err.msg)
        # An unfixed adaptive kernel
        prob = Model(Union{ContaminatedGaussian{Float64}, Float64}, TlAdapt)
        addvariable!(prob, ContaminatedGaussian(1.0, 5.0, 0.8))
        addvariable!(prob, 2.0)
        addcost!(prob, TlAdapt(SVector(1, 2)))
        err = try optimize!(prob, TallOptions; formulation=:tall); nothing catch e e end
        @test err isa ArgumentError
        @test occursin("tall", err.msg) && occursin("adaptive", err.msg)
        # The same models are fine on the normal equations' own terms (the error is the tall formulation's)
        @test optimize!(prob, TallOptions; unfixed=2, numthreads=static(1)) isa OptimizationResult
    end

    @testset "too large for a dense Jacobian: throws before allocating" begin
        # 20000 residuals by 3 variables: 480 kB if dense
        model = curvefit(n=20000)
        @test_throws ArgumentError optimize!(deepcopy(model), TallOptions; formulation=:tall, max_dense_bytes=1000)
        err = try optimize!(deepcopy(model), TallOptions; formulation=:tall, max_dense_bytes=1000); nothing catch e e end
        @test occursin("20000×3", err.msg) && occursin("dense", err.msg) && occursin("1000", err.msg)
        m = deepcopy(model)
        alloc = @allocated try optimize!(m, TallOptions; formulation=:tall, max_dense_bytes=1000) catch end
        @test alloc < 50_000
        # The limit counts only the unfixed variables, and allows exactly its size
        big = posemodel()
        n = 8 * NLLSModel.nresiduals(big) * 9 # 3 + 3 + 2 + 1 degrees of freedom
        @test_throws ArgumentError optimize!(deepcopy(big), TallOptions; formulation=:tall, numthreads=static(1), max_dense_bytes=n - 1)
        @test optimize!(deepcopy(big), TallOptions; formulation=:tall, numthreads=static(1), max_dense_bytes=n - 1, unfixed=[1, 2]) isa OptimizationResult
        @test optimize!(deepcopy(big), TallOptions; formulation=:tall, numthreads=static(1), max_dense_bytes=n) isa OptimizationResult
    end

    @testset "rejected by the optimizer" begin
        model = curvefit()
        for options in (OptimizationOptions(iterator=:gradientdescent), OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:cholesky))
            err = try optimize!(deepcopy(model), options; formulation=:tall, numthreads=static(1)); nothing catch e e end
            @test err isa ArgumentError
        end
        # They are fine for the normal equations
        @test optimize!(deepcopy(model), OptimizationOptions(iterator=:gradientdescent); numthreads=static(1)) isa OptimizationResult
    end

    # Z3: iterators that linearize at the first trial evaluate resjac! at a point that may then be rejected; with the optimizer's
    # current point being model.variables itself, that must not disturb it
    @testset "rejected trials with $iterator reach the optimum of :normal" for (iterator, expect_rejection) in ((:levenbergmarquardt, true), (:levenbergmarquardt_dtd, true), (:dogleg, true), (:newton, false))
        make() = (m = curvefit(outliers=2); m.variables[1] = [0.1, 6.0, -4.0]; m)
        options = OptimizationOptions(iterator=iterator, maxiters=300)
        normal, rn = optimize(make(), options; numthreads=static(1))
        model = make()
        start = deepcopy(model.variables)
        tall, rt = optimize(model, options; formulation=:tall, numthreads=static(1))
        # A trial that is evaluated by f rather than by the first-trial linearization has been rejected once
        expect_rejection && @test rt.costcomputations > 0
        @test model.variables == start
        @test NLLSModel.cost(tall) ≈ rt.bestcost rtol=1e-12
        @test NLLSModel.cost(tall) ≈ NLLSModel.cost(normal) rtol=1e-8
        # With the variable alone in a list of variables (so x0 is model.variables) and the normal one
        multi = Model(Vector{Float64}, Union{TlExp, Tgt{Vector{Float64}}})
        base = make()
        addvariable!(multi, base.variables[1]); addvariable!(multi, [0.7])
        foreach(c -> addcost!(multi, TlExp(c.t, c.y, 1)), costlist(base))
        addcost!(multi, Tgt([0.2], 2))
        nn, _ = optimize(multi, options; numthreads=static(1))
        rr = optimize!(multi, options; formulation=:tall, numthreads=static(1))
        @test NLLSModel.cost(multi) ≈ rr.bestcost rtol=1e-12
        @test NLLSModel.cost(multi) ≈ NLLSModel.cost(nn) rtol=1e-8
        expect_rejection && @test rr.costcomputations > 0
    end

    @testset "optimize leaves the original model unchanged" begin
        for make in (curvefit, rotproblem, posemodel)
            model = make()
            start = deepcopy(model.variables)
            c0 = NLLSModel.cost(model)
            solved, result = optimize(model, TallOptions; formulation=:tall, numthreads=static(1))
            @test model.variables == start # Bit for bit
            @test NLLSModel.cost(model) == c0
            @test solved !== model
            @test NLLSModel.cost(solved) ≈ result.bestcost atol=1e-12
            @test NLLSModel.cost(solved) < c0
        end
        model, _ = singleproblem(SVector{3, Float64}, 3, 9)
        start = model.variables[1]
        solved, result = optimize(model, TallOptions; formulation=:tall)
        @test model.variables[1] === start
    end

    @testset "threads: the same result however many" begin
        model = posemodel()
        a, ra = optimize(model, TallOptions; formulation=:tall, numthreads=static(1), min_work_per_thread=1)
        b, rb = optimize(model, TallOptions; formulation=:tall, numthreads=static(Threads.nthreads()), min_work_per_thread=1)
        @test NLLSModel.cost(a) ≈ NLLSModel.cost(b) rtol=1e-10
        @test samevars(a, b; atol=1e-8)
    end

    @testset "the optimizer's own types satisfy the evaluator (T3)" begin
        model = posemodel()
        f, _, resjac!, x0, J = tlnls(model; numthreads=static(1), matrix_type=:dense)
        w = TallExt.PrezeroedResJac(resjac!)
        m = size(J, 1)
        @test J isa Matrix{Float64}
        r = zeros(m) # The optimizer's own buffers: a Vector{Float64} and a Matrix{Float64}
        Jb = zeros(m, size(J, 2))
        c = w(r, Jb, x0)
        # The wrapper does not zero, as the optimizer has already: garbage outside the Jacobian's structure is left alone
        Jg1, Jg2 = fill(1.0, size(Jb)), fill(2.0, size(Jb))
        @test w(zeros(m), Jg1, x0) == c
        w(zeros(m), Jg2, x0)
        untouched = (Jg1 .== 1.0) .& (Jg2 .== 2.0)
        @test any(untouched)
        @test Jg1[.!untouched] ≈ Jb[.!untouched]
        @test all(iszero, Jb[untouched])
        # The evaluator itself zeroes unless told not to
        Jz = fill(1.0, size(Jb))
        @test resjac!(zeros(m), Jz, x0) == c
        @test Jz == Jb
        @test (@inferred TallExt.PrezeroedResJac{typeof(resjac!)}(resjac!)) isa TallExt.PrezeroedResJac
        @test isconcretetype(typeof(w))
    end
end
