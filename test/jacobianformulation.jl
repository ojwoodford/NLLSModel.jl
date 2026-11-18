using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars

include("fixtures/loader.jl")
fixture("frontdoor", "jacobianformulation", "jacobianproblems")

# optimize!(model, options; formulation = :jacobian): the residual-and-Jacobian path.

# --- Tests ------------------------------------------------------------------------------------------------------------

@testset "jacobianformulation.jl" begin
    @test JacobianExt isa Module

    @testset "formulation is a keyword of optimize! only" begin
        @test !(:formulation in fieldnames(typeof(JacobianOptions)))
        model = curvefit()
        @test_throws ArgumentError optimize!(model, JacobianOptions; formulation=:sparse)
        # The values before the rename are refused like any unknown value: no alias (rename N1). They are spelt in
        # pieces so that a search of the package for the old names finds none
        @testset "formulation = $(repr(old)) is refused" for old in (Symbol("nor", "mal"), Symbol("ta", "ll"))
            err = try optimize!(deepcopy(model), JacobianOptions; formulation=old); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("expected :hessian or :jacobian", err.msg)
        end
        @test_throws ArgumentError optimize!(model, JacobianOptions; formulation=:jacobian, reduction=:bogus)
        # Not inferred from the linear solver: asking for the QR solver on the Hessian formulation is the optimizer's business
        r = optimize!(deepcopy(model), OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:qr, maxiters=200); formulation=:hessian, numthreads=static(1))
        @test r isa OptimizationResult
        # :hessian is the default
        a, ra = optimize(model, JacobianOptions; numthreads=static(1))
        b, rb = optimize(model, JacobianOptions; formulation=:hessian, numthreads=static(1))
        @test a.variables == b.variables
        @test ra.bestcost == rb.bestcost
    end

    @testset "same optimum as :hessian: $name" for (name, make) in (
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

    @testset "same optimum as :hessian: a single static variable of size 3 (through optimize)" begin
        model, t = singleproblem(SVector{3, Float64}, 3, 7)
        ok, _, _ = sameoptimum(model)
        @test ok
        solved, result = optimize(model, JacobianOptions; formulation=:jacobian)
        @test solved.variables[1] ≈ t atol=1e-6
        @test NLLSModel.cost(solved) ≈ result.bestcost atol=1e-12
        # optimize! of a single static variable too
        m2, t2 = singleproblem(SVector{3, Float64}, 3, 8)
        r = optimize!(m2, JacobianOptions; formulation=:jacobian)
        @test m2.variables[1] ≈ t2 atol=1e-6
        @test NLLSModel.cost(m2) ≈ r.bestcost atol=1e-12
    end

    # NTP3: a small single static variable gets the optimizer's static Jacobian system, which hands resjac! an MMatrix copy of
    # the prototype (and an MVector for r); the evaluators take it as they take the Matrix
    @testset "a single static variable: the optimizer's static system ($name)" for (name, kernel) in (("no kernel", nothing),
            ("Huber2o", Huber2oKernel(0.3)))
        model, t = singleproblem(SVector{3, Float64}, 3, 7)
        if kernel !== nothing
            robust = Model(SVector{3, Float64}, TlRob{Tgt{SVector{3, Float64}}, typeof(kernel)})
            addvariable!(robust, model.variables[1])
            addcost!(robust, TlRob(Tgt(t, 1), kernel))
            model = robust
        end
        f, _, rj, x0, J = tlnls(model; numthreads=static(1))
        @test x0 isa SVector{3, Float64} && J isa Matrix{Float64} # 3 × 3: under the optimizer's static cap
        # As the front door builds it, with a spy on the arguments resjac! receives
        seen = Set{Any}()
        spy = (r, J, x) -> (push!(seen, (typeof(r), typeof(J))); JacobianExt.JacobianResJac(rj)(r, J, x))
        x, res = SecondOrderOptimization.optimize(x0, JacobianObjective(f, spy, J), JacobianOptions)
        @test seen == Set([(MVector{3, Float64}, MMatrix{3, 3, Float64, 9})])
        # The front door reaches the Hessian path's cost and the spy's point
        solved, rt = optimize(model, JacobianOptions; formulation=:jacobian)
        solved_hes, _ = optimize(model, JacobianOptions)
        @test NLLSModel.cost(solved) ≈ NLLSModel.cost(solved_hes) rtol=1e-8 atol=1e-14
        @test solved.variables[1] ≈ x rtol=1e-12
        @test rt.bestcost ≈ res.bestcost rtol=1e-12 atol=1e-20
    end

    @testset "same optimum, robust Huber with outliers is not the least-squares optimum" begin
        # Guard against the Huber tests above passing because the kernel was silently ignored
        ls = optimize(curvefit(outliers=4), JacobianOptions; formulation=:jacobian, numthreads=static(1))[1]
        hub = optimize(curvefit(huber=true, outliers=4), JacobianOptions; formulation=:jacobian, numthreads=static(1))[1]
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
                result = optimize!(model, OptimizationOptions(iterator=:levenbergmarquardt, maxiters=maxiters); formulation=:jacobian, numthreads=static(1), callback=leavejunk, unfixed)
                @test NLLSModel.cost(model) ≈ result.bestcost rtol=1e-12
                k = unfixed isa Vector ? unfixed[1] : something(unfixed, 1)
                @test model.variables[k] != lastjunk[]
            end
        end
        # Without it the model would be left on the last point evaluated: run the optimizer on the evaluators directly
        m = posemodel()
        f, _, resjac!, x0, J = tlnls(m; unfixed=[1, 3], numthreads=static(1), storage=:dense)
        obj = JacobianObjective(f, JacobianExt.JacobianResJac(resjac!), J)
        result = SecondOrderOptimization.optimize!(x0, obj, JacobianOptions)
        resjac!(zeros(size(J, 1)), J, update!(deepcopy(x0), x0, fill(0.3, 4))) # A rejected trial, the last thing evaluated
        @test !(NLLSModel.cost(m) ≈ result.bestcost)
        syncvars!(m, resjac!, x0)
        @test NLLSModel.cost(m) ≈ result.bestcost rtol=1e-12
        # Each unfixed layout: one variable, a list, and all
        for unfixed in (3, 2, [1, 4], nothing)
            m = posemodel()
            f, res!, resjac!, x0, J = tlnls(m; unfixed, numthreads=static(1), storage=:dense)
            x = update!(deepcopy(x0), x0, fill(0.1, size(J, 2)))
            for ev in (res!, resjac!)
                resjac!(zeros(size(J, 1)), J, x0) # The model holds x0
                syncvars!(m, ev, x)
                f2, = tlnls(m; unfixed, numthreads=static(1), storage=:dense)
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
        result = optimize!(model, JacobianOptions; formulation=:jacobian, unfixed, numthreads=static(1))
        for k in setdiff(1:4, free)
            @test model.variables[k] == start[k] # Fixed variables are untouched, bit for bit
        end
        # The free variables are at the optimum: the same as the Hessian formulation finds
        ref = deepcopy(posemodel())
        optimize!(ref, JacobianOptions; unfixed, numthreads=static(1))
        @test NLLSModel.cost(model) ≈ NLLSModel.cost(ref) rtol=1e-8
        @test NLLSModel.cost(model) ≈ result.bestcost atol=1e-12
        for k in free
            @test isapprox(model.variables[k], ref.variables[k]; atol=1e-5)
        end
        @test any(k -> model.variables[k] != start[k], free)
    end

    @testset "keywords: reduction" begin
        # One variable: never partitioned, on either path, whatever the reduction (NTP2)
        model = curvefit()
        for reduction in (:auto, :none, :eliminate)
            @test optimize!(deepcopy(model), JacobianOptions; formulation=:jacobian, reduction, numthreads=static(1)) isa OptimizationResult
        end
        # Several variables, none eliminable: :eliminate is the partition's error, from optimvars_nls, prefixed by the front door
        err = try optimize!(posemodel(), JacobianOptions; formulation=:jacobian, reduction=:eliminate); nothing catch e e end
        @test err isa ArgumentError
        @test startswith(err.msg, "formulation = :jacobian:") && occursin("eliminable", err.msg)
        # and the Hessian formulation still takes all three
        @test optimize!(posemodel(), JacobianOptions; reduction=:none, numthreads=static(1)) isa OptimizationResult
    end

    # NTP1: the front door hands over the Jacobian optimvars_nls chooses, dense or sparse
    @testset "keywords: storage on :jacobian, $name" for (name, make) in (
            ("curve fit, one Vector", () -> curvefit()),
            ("several variables of different sizes", () -> first(mixedproblem())),
            ("chain of Vectors", () -> first(chainproblem(6, 3, 5))),
            ("rotation, Vector, SVector, Vector", () -> posemodel()),
            ("Huber on every cost", () -> curvefit(huber=true, outliers=4)))
        model = make()
        c0 = NLLSModel.cost(model)
        solved_hes, _ = optimize(model, JacobianOptions; numthreads=static(1))
        dense, rd = optimize(model, JacobianOptions; formulation=:jacobian, numthreads=static(1), storage=:dense)
        sparse, rs = optimize(model, JacobianOptions; formulation=:jacobian, numthreads=static(1), storage=:sparse)
        cn, cd, cs = NLLSModel.cost(solved_hes), NLLSModel.cost(dense), NLLSModel.cost(sparse)
        @test cs ≈ cd rtol=1e-8 atol=1e-10 * c0
        @test cs ≈ cn rtol=1e-8 atol=1e-10 * c0
        @test cs ≈ rs.bestcost rtol=1e-12
        @test samevars(sparse, dense; atol=1e-5)
    end

    @testset "keywords: storage = :auto on :jacobian keeps optimvars_nls's choice" begin
        small = curvefit()
        big, _ = chainproblem(200, 3, 4) # 1 197 x 600, under 1% filled
        # The layout's choice, unchanged: dense for a small Jacobian, sparse for a large one with little fill
        @test tlnls(small; numthreads=static(1))[5] isa Matrix{Float64}
        @test tlnls(big; numthreads=static(1))[5] isa SparseMatrixCSC{Float64, Int}
        # The front door hands that choice over: the large one is solved sparse, under a cap too small for it dense
        cap = 8 * 1197 * 600 - 1
        solved_jac, rt = optimize(big, JacobianOptions; formulation=:jacobian, numthreads=static(1), max_dense_bytes=cap)
        solved_hes, _ = optimize(big, JacobianOptions; numthreads=static(1))
        @test NLLSModel.cost(solved_jac) ≈ NLLSModel.cost(solved_hes) rtol=1e-8 atol=1e-10 * NLLSModel.cost(big)
        # Forced dense above the cap: an ArgumentError before anything is allocated, prefixed by the front door
        err = try optimize!(deepcopy(big), JacobianOptions; formulation=:jacobian, numthreads=static(1), storage=:dense, max_dense_bytes=cap); nothing catch e e end
        @test err isa ArgumentError && startswith(err.msg, "formulation = :jacobian:") && occursin("$(8 * 1197 * 600) bytes", err.msg)
        @test_logs optimize!(deepcopy(small), JacobianOptions; formulation=:jacobian, numthreads=static(1), storage=:sparse)
    end

    @testset "kernels (R1): the kernel decides, there is no robust keyword" begin
        @test_throws MethodError optimize!(curvefit(), JacobianOptions; formulation=:jacobian, robust=:irls)
        @test_throws MethodError optimize!(curvefit(), JacobianOptions; robust=:none)
        # NoRobust, Huber and Huber2o costs in one model reach the same optimum as the Hessian formulation
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
        @test norm(optimize(model, JacobianOptions; formulation=:jacobian)[1].variables[1] - optimize(base, JacobianOptions; formulation=:jacobian)[1].variables[1]) > 1e-3
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
        out, res = optimize(prob, JacobianOptions; formulation=:jacobian, unfixed=2, numthreads=static(1))
        @test out.variables[2] ≈ 1.0 atol=1e-6
    end

    @testset "unsupported models fail clearly, naming the Jacobian formulation" begin
        # A cost that is not a residual
        model = Model(Vector{Float64}, TlCost)
        addvariable!(model, [1.0, 2.0])
        addcost!(model, TlCost(1))
        err = try optimize!(model, JacobianOptions; formulation=:jacobian); nothing catch e e end
        @test err isa ArgumentError
        @test occursin("formulation = :jacobian", err.msg) && occursin("AbstractResidual", err.msg)
        # An unfixed adaptive kernel
        prob = Model(Union{ContaminatedGaussian{Float64}, Float64}, TlAdapt)
        addvariable!(prob, ContaminatedGaussian(1.0, 5.0, 0.8))
        addvariable!(prob, 2.0)
        addcost!(prob, TlAdapt(SVector(1, 2)))
        err = try optimize!(prob, JacobianOptions; formulation=:jacobian); nothing catch e e end
        @test err isa ArgumentError
        @test occursin("formulation = :jacobian", err.msg) && occursin("adaptive", err.msg)
        # The same models are fine on the Hessian formulation's own terms (the error is the Jacobian formulation's)
        @test optimize!(prob, JacobianOptions; unfixed=2, numthreads=static(1)) isa OptimizationResult
    end

    @testset "too large for a dense Jacobian: storage = :dense throws before allocating, :auto goes sparse" begin
        # 20000 residuals by 3 variables: 480 kB if dense
        model = curvefit(n=20000)
        @test_throws ArgumentError optimize!(deepcopy(model), JacobianOptions; formulation=:jacobian, storage=:dense, max_dense_bytes=1000)
        err = try optimize!(deepcopy(model), JacobianOptions; formulation=:jacobian, storage=:dense, max_dense_bytes=1000); nothing catch e e end
        @test occursin("20000×3", err.msg) && occursin("dense", err.msg) && occursin("1000", err.msg) && occursin("formulation = :jacobian", err.msg)
        m = deepcopy(model)
        alloc = @allocated try optimize!(m, JacobianOptions; formulation=:jacobian, storage=:dense, max_dense_bytes=1000) catch end
        @test alloc < 50_000
        # Under :auto the same cap gives a sparse Jacobian, and the solve goes ahead
        @test tlnls(model; numthreads=static(1), max_dense_bytes=1000)[5] isa SparseMatrixCSC
        @test optimize!(deepcopy(model), JacobianOptions; formulation=:jacobian, numthreads=static(1), max_dense_bytes=1000) isa OptimizationResult
        # The limit counts only the unfixed variables, and allows exactly its size
        big = posemodel()
        n = 8 * NLLSModel.nresiduals(big) * 9 # 3 + 3 + 2 + 1 degrees of freedom
        @test_throws ArgumentError optimize!(deepcopy(big), JacobianOptions; formulation=:jacobian, numthreads=static(1), storage=:dense, max_dense_bytes=n - 1)
        @test optimize!(deepcopy(big), JacobianOptions; formulation=:jacobian, numthreads=static(1), storage=:dense, max_dense_bytes=n - 1, unfixed=[1, 2]) isa OptimizationResult
        @test optimize!(deepcopy(big), JacobianOptions; formulation=:jacobian, numthreads=static(1), storage=:dense, max_dense_bytes=n) isa OptimizationResult
    end

    @testset "a linsolver the optimizer rejects; gradient descent it accepts" begin
        model = curvefit()
        # A linsolver not in the table: :cholesky with a plain Jacobian
        options = OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:cholesky)
        err = try optimize!(deepcopy(model), options; formulation=:jacobian, numthreads=static(1)); nothing catch e e end
        @test err isa ArgumentError
        # It is fine for the Hessian formulation
        @test optimize!(deepcopy(model), options; numthreads=static(1)) isa OptimizationResult
        # Gradient descent is accepted on the Jacobian path (SecondOrderOptimization IT2), as on the Hessian one
        c0 = NLLSModel.cost(model)
        for formulation in (:jacobian, :hessian)
            m = deepcopy(model)
            @test optimize!(m, OptimizationOptions(iterator=:gradientdescent); formulation, numthreads=static(1)) isa OptimizationResult
            @test NLLSModel.cost(m) < c0
        end
    end

    # Z3: iterators that linearize at the first trial evaluate resjac! at a point that may then be rejected; with the optimizer's
    # current point being model.variables itself, that must not disturb it
    @testset "rejected trials with $iterator reach the optimum of :hessian" for (iterator, expect_rejection) in ((:levenbergmarquardt, true), (:levenbergmarquardt_dtd, true), (:dogleg, true), (:newton, false))
        make() = (m = curvefit(outliers=2); m.variables[1] = [0.1, 6.0, -4.0]; m)
        options = OptimizationOptions(iterator=iterator, maxiters=300)
        solved_hes, rn = optimize(make(), options; numthreads=static(1))
        model = make()
        start = deepcopy(model.variables)
        solved_jac, rt = optimize(model, options; formulation=:jacobian, numthreads=static(1))
        # A trial that is evaluated by f rather than by the first-trial linearization has been rejected once
        expect_rejection && @test rt.costcomputations > 0
        @test model.variables == start
        @test NLLSModel.cost(solved_jac) ≈ rt.bestcost rtol=1e-12
        @test NLLSModel.cost(solved_jac) ≈ NLLSModel.cost(solved_hes) rtol=1e-8
        # With the variable alone in a list of variables (so x0 is model.variables) and the Hessian one
        multi = Model(Vector{Float64}, Union{TlExp, Tgt{Vector{Float64}}})
        base = make()
        addvariable!(multi, base.variables[1]); addvariable!(multi, [0.7])
        foreach(c -> addcost!(multi, TlExp(c.t, c.y, 1)), costlist(base))
        addcost!(multi, Tgt([0.2], 2))
        nn, _ = optimize(multi, options; numthreads=static(1))
        rr = optimize!(multi, options; formulation=:jacobian, numthreads=static(1))
        @test NLLSModel.cost(multi) ≈ rr.bestcost rtol=1e-12
        @test NLLSModel.cost(multi) ≈ NLLSModel.cost(nn) rtol=1e-8
        expect_rejection && @test rr.costcomputations > 0
    end

    @testset "optimize leaves the original model unchanged" begin
        for make in (curvefit, rotproblem, posemodel)
            model = make()
            start = deepcopy(model.variables)
            c0 = NLLSModel.cost(model)
            solved, result = optimize(model, JacobianOptions; formulation=:jacobian, numthreads=static(1))
            @test model.variables == start # Bit for bit
            @test NLLSModel.cost(model) == c0
            @test solved !== model
            @test NLLSModel.cost(solved) ≈ result.bestcost atol=1e-12
            @test NLLSModel.cost(solved) < c0
        end
        model, _ = singleproblem(SVector{3, Float64}, 3, 9)
        start = model.variables[1]
        solved, result = optimize(model, JacobianOptions; formulation=:jacobian)
        @test model.variables[1] === start
    end

    @testset "the optimizer's own types satisfy the evaluator (T3)" begin
        model = posemodel()
        f, _, resjac!, x0, J = tlnls(model; numthreads=static(1), storage=:dense)
        w = JacobianExt.JacobianResJac(resjac!)
        m = size(J, 1)
        @test J isa Matrix{Float64}
        r = zeros(m) # The optimizer's own buffers: a Vector{Float64} and a Matrix{Float64}
        Jb = zeros(m, size(J, 2))
        c = w(r, Jb, x0)
        # The wrapper does not fill J first (prezeroed = true: the structural zeros are in place): garbage outside the Jacobian's
        # structure is left alone
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
        @test (@inferred JacobianExt.JacobianResJac{typeof(resjac!)}(resjac!)) isa JacobianExt.JacobianResJac
        @test isconcretetype(typeof(w))
    end
end
