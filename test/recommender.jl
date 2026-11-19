using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian
import DampedLinearSolve

include("fixtures/loader.jl")
fixture("recommender")

# RC3: the recommender's features, rules and choices; RC4: how it is displayed (src/recommender.jl)

@testset "recommender.jl" begin
    @testset "rule L, the Hessian formulation's linear solver" begin
        for kind in (:static, :dense, :schur_dense, :schur_sparse)
            @test NLLSModel.rule_linsolver(true, kind)[1] === :cholesky
        end
        @test NLLSModel.rule_linsolver(true, :static) == (:cholesky, :L_static)
        @test NLLSModel.rule_linsolver(true, :dense) == (:cholesky, :L_dense)
        @test NLLSModel.rule_linsolver(true, :schur_sparse) == (:cholesky, :L_schur)
        @test NLLSModel.rule_linsolver(true, :sparse) == (:ldl, :L_sparse)
        for kind in (:static, :dense, :sparse, :schur_dense, :schur_sparse)
            @test NLLSModel.rule_linsolver(false, kind) == (:ldl, :L_nonresidual)
        end
    end

    @testset "rule F, the formulation" begin
        F(; kw...) = NLLSModel.rule_formulation(recfeatures(; kw...))
        @test F(condition=NLLSModel.REC_JACOBIAN_CONDITION) == (:jacobian, :F_illconditioned)
        @test F(condition=prevfloat(NLLSModel.REC_JACOBIAN_CONDITION)) == (:hessian, :F_wellconditioned)
        @test F(condition=Inf, status=DampedLinearSolve.NotPositiveDefinite, ranktest=:nofactorization) == (:hessian, :F_singular)
        @test F(condition=1e20, allresiduals=false) == (:hessian, :F_nonresidual)
        @test F(condition=Inf, allresiduals=false) == (:hessian, :F_nonresidual)
    end

    @testset "rule F, the rank test (RC8)" begin
        F(; kw...) = NLLSModel.rule_formulation(recfeatures(; kw...))
        tol = NLLSModel.REC_RANK_TOLERANCE
        # An ill-conditioned Hessian and an undamped failure take the same branches, the tolerance from both sides
        for ill in ((condition=1e15,), (condition=Inf, status=DampedLinearSolve.NotPositiveDefinite),
                (condition=Inf, status=DampedLinearSolve.Indefinite))
            @test NLLSModel.needsranktest(recfeatures(; ill...))
            @test F(; ill..., rank_t=tol) == (:jacobian, :F_illconditioned)
            @test F(; ill..., rank_t=prevfloat(tol)) == (:hessian, :F_rankdeficient)
            @test F(; ill..., rank_t=0.0) == (:hessian, :F_rankdeficient)
            @test F(; ill..., ranktest=:nodiagonal) == (:hessian, :F_singular)
            @test F(; ill..., ranktest=:nofactorization) == (:hessian, :F_singular)
            @test F(; ill..., ranktest=:nojacobian) == (:hessian, :F_jacobian_unavailable)
            @test F(; ill..., allresiduals=false) == (:hessian, :F_nonresidual)
        end
        # Not needed: a well-conditioned Hessian that factorizes undamped, or a cost that is not a residual
        @test !NLLSModel.needsranktest(recfeatures(condition=prevfloat(NLLSModel.REC_JACOBIAN_CONDITION)))
        @test NLLSModel.needsranktest(recfeatures(condition=NLLSModel.REC_JACOBIAN_CONDITION))
        @test !NLLSModel.needsranktest(recfeatures(condition=1e20, allresiduals=false))
        # A failed undamped solve needs the test, whatever the condition field says
        @test NLLSModel.needsranktest(recfeatures(condition=10.0, status=DampedLinearSolve.NotPositiveDefinite))
    end

    @testset "rule I, the iterator" begin
        I_(; kw...) = NLLSModel.rule_iterator(recfeatures(; kw...))
        @test I_() == (:dogleg, :I_dogleg)
        # Each of dogleg's tests, from both sides
        @test I_(condition=prevfloat(NLLSModel.REC_DOGLEG_CONDITION)) == (:dogleg, :I_dogleg)
        @test I_(condition=NLLSModel.REC_DOGLEG_CONDITION) == (:levenbergmarquardt, :I_lm)
        @test I_(diagspread=prevfloat(NLLSModel.REC_DIAGSPREAD)) == (:dogleg, :I_dogleg)
        @test I_(diagspread=NLLSModel.REC_DIAGSPREAD) == (:levenbergmarquardt_dtd, :I_dtd)
        @test I_(t_solve=1e-5, t_linearize=1e-5) == (:dogleg, :I_dogleg)
        @test I_(t_solve=prevfloat(1e-5), t_linearize=1e-5) == (:levenbergmarquardt, :I_lm)
        @test I_(status=DampedLinearSolve.Indefinite) == (:levenbergmarquardt, :I_lm)
        @test I_(status=DampedLinearSolve.NotPositiveDefinite, condition=Inf) == (:levenbergmarquardt, :I_lm)
        # A spread diagonal: D'D scaling, whatever else holds
        @test I_(diagspread=Inf, status=DampedLinearSolve.NotPositiveDefinite, condition=Inf) == (:levenbergmarquardt_dtd, :I_dtd)
        @test I_(diagspread=prevfloat(NLLSModel.REC_DIAGSPREAD), condition=Inf) == (:levenbergmarquardt, :I_lm)
    end

    @testset "features and choices: $(spec.name)" for spec in REC_MODELS
        model = spec.make()
        start = deepcopy(model.variables)
        rec = NLLSModel.recommendoptimizer(model; numthreads=static(1))
        @test rec isa NLLSModel.OptimizationRecommendation
        @test model.variables == start # The model is not changed
        f = rec.features
        @test f.hessiansystem === spec.kind
        @test rec.formulation === spec.formulation
        @test rec.linsolver === spec.linsolver
        @test f.nvariables == length(model.variables)
        @test f.dimension == sum(v -> Int(dynamic(OptimizationVariableInterface.nvars(v))), model.variables)
        @test f.ncosts == length(collect(Iterators.flatten(values(model.costs))))
        @test f.allresiduals == (spec.name != "a non-residual cost")
        @test f.nresiduals == sum(c -> c isa NLLSModel.AbstractResidual ? Int(dynamic(NLLSModel.nres(c))) : 0, Iterators.flatten(values(model.costs)); init=0)
        @test (f.retained > 0) == (f.eliminated > 0) == (spec.kind in (:schur_dense, :schur_sparse))
        f.retained > 0 && @test f.retained + f.eliminated == f.dimension
        @test f.diag_min <= f.diag_max
        @test f.diagspread == (f.diag_min > 0 ? f.diag_max / f.diag_min : Inf)
        @test isfinite(f.condition) == (f.status == DampedLinearSolve.Success)
        @test all(t -> t >= 0, (f.t_cost, f.t_linearize, f.t_solve))
        @test rec.unfixed === nothing && rec.numthreads === static(1) && rec.max_dense_bytes == NLLSModel.MAX_DENSE_BYTES
        @test NLLSModel.rule_formulation(f) == (rec.formulation, rec.rules.formulation)
        @test (f.ranktest !== :notneeded) == NLLSModel.needsranktest(f)
        @test isnan(f.rank_t) == isnan(f.rank_damping) == (f.ranktest !== :ran)
        @test rec.iterator === NLLSModel.rule_iterator(f)[1]

        # reduction and storage are what optimvars (optimvars_nls on the Jacobian path) builds with :auto
        if rec.formulation === :hessian
            @test rec.rules.reduction === rec.rules.storage === :RS_hessian
            @test (rec.reduction, rec.storage) == NLLSModel.reductionstorage(NLLSModel.hessiansystemkind(optimvars(model; numthreads=static(1))[4]))
        else
            @test rec.rules.reduction === rec.rules.storage === :RS_jacobian
            @test rec.rules.linsolver === :L_jacobian
            @test (rec.reduction, rec.storage) == NLLSModel.jacobianreductionstorage(optimvars_nls(model; numthreads=static(1))[5])
        end
        @test (rec.reduction === :eliminate) == (spec.kind in (:schur_dense, :schur_sparse))

        # Two calls agree on everything but the timings (and the iterator, where the timings decided it)
        rec2 = NLLSModel.recommendoptimizer(model; numthreads=static(1))
        @test isequal(untimed(rec2.features), untimed(f))
        @test (rec2.formulation, rec2.reduction, rec2.storage, rec2.linsolver) == (rec.formulation, rec.reduction, rec.storage, rec.linsolver)
        @test rec2.fallback == rec.fallback
        timingdecided(f) || @test rec2.iterator === rec.iterator
        @test model.variables == start
    end

    @testset "the fallback from the Jacobian path" begin
        rec = NLLSModel.recommendoptimizer(adaptivemodel(); numthreads=static(1))
        @test rec.features.condition >= NLLSModel.REC_JACOBIAN_CONDITION
        @test rec.formulation === :hessian && rec.rules.formulation === :F_jacobian_unavailable
        @test occursin("adaptive", rec.fallback)
        @test rec.rules.reduction === :RS_hessian && rec.rules.linsolver === :L_dense
    end

    @testset "the rank test: $(spec.name)" for spec in REC_RANK_MODELS
        model = spec.make()
        start = deepcopy(model.variables)
        rec = NLLSModel.recommendoptimizer(model; numthreads=static(1))
        f = rec.features
        @test model.variables == start # The model is not changed
        @test NLLSModel.needsranktest(f)
        @test rec.rules.formulation === spec.rule && f.ranktest === spec.ranktest
        @test (f.hessiansystem === :static) == occursin("static", spec.name)
        @test rec.formulation === (spec.rule === :F_illconditioned ? :jacobian : :hessian)
        @test isequal(f.rank_damping, spec.damping)
        @test (f.status == DampedLinearSolve.Success) == (spec.damping == 0) # Damping only when the undamped solve failed
        @test isfinite(f.condition) == (f.status == DampedLinearSolve.Success)
        if rec.formulation === :jacobian
            @test rec.linsolver === :qr && rec.rules.linsolver === :L_jacobian && rec.rules.reduction === :RS_jacobian
            @test (rec.reduction, rec.storage) == NLLSModel.jacobianreductionstorage(optimvars_nls(model; numthreads=static(1))[5])
        end
        # Two calls agree on everything but the timings
        rec2 = NLLSModel.recommendoptimizer(model; numthreads=static(1))
        @test isequal(untimed(rec2.features), untimed(f))
        @test (rec2.formulation, rec2.reduction, rec2.storage, rec2.linsolver) == (rec.formulation, rec.reduction, rec.storage, rec.linsolver)
        @test model.variables == start
        # The explanation, and the call it prints runs
        @test occursin(flat(NLLSModel.formulation_reason(rec)), flat(longform(rec)))
        @test all(line -> textwidth(line) <= 80, split(longform(rec), '\n'))
        positional, keywords, optkw = printedcall(rec)
        @test keywords[:formulation] == QuoteNode(rec.formulation) && optkw[:linsolver] == QuoteNode(rec.linsolver)
        # The printed call runs, and does not raise the cost
        copied = deepcopy(model)
        runprintedcall(rec, copied)
        @test cost(copied) <= cost(model)
    end

    @testset "the rank test's readings" begin
        tol = NLLSModel.REC_RANK_TOLERANCE
        # Full rank, ill-conditioned, factorizes undamped: t is an upper bound on the smallest singular value of the
        # column-scaled J, and close to it (measured: equal to three digits on Julia 1.13 and 1.10)
        model = REC_RANK_MODELS[1].make()
        t = NLLSModel.recommendoptimizer(model; numthreads=static(1)).features.rank_t
        sv = scaledsvdvals(rankpieces(model).J)
        σ = last(sv)
        @test 1e7 < first(sv) / σ < 1e8 # The condition number of the column-scaled J
        @test σ * (1 - 1e-8) <= t <= 1.05 * σ
        # Without and with the refinement step; the test reads the smaller
        r = rankreadings(model)
        @test length(r) == NLLSModel.REC_RANK_REFINEMENTS + 1 && t == minimum(r)
        # A null space with a remainder to 1e-4 and to 1e-6: rank-deficient after the refinement. To 1e-6 the unrefined
        # reading is above the tolerance, so the refinement decides (measured on Julia 1.13: 7.3e-14 and 1.8e-16 to 1e-4,
        # 4.1e-11 and 1.9e-14 to 1e-6)
        r4 = rankreadings(REC_RANK_MODELS[3].make())
        r6 = rankreadings(REC_RANK_MODELS[4].make())
        @test r4[2] < tol && r6[2] < tol
        @test r6[1] >= tol
        @test r4[2] < r4[1] && r6[2] < r6[1]
        # The degree-14 monomial fit: full rank, read through the damping (measured: 71 and 69 times the true value, 2.5e-10,
        # on Julia 1.13 and 1.10)
        model = REC_RANK_MODELS[2].make()
        t = NLLSModel.recommendoptimizer(model; numthreads=static(1)).features.rank_t
        σ = scaledsigmamin(rankpieces(model).J)
        @test σ <= t <= 200 * σ
        # On a J of full rank the refinement step is zero but for rounding, so its reading can be far above the unrefined
        # one; the smaller is kept (the existing fixture "ill-conditioned, one variable", measured: 3.2e-7 unrefined, and
        # 0.066 refined on Julia 1.13, 0.45 on 1.10)
        model = linearmodel(scaledjacobian(3, 1 - 1e-13))
        r = rankreadings(model)
        @test NLLSModel.recommendoptimizer(model; numthreads=static(1)).features.rank_t == minimum(r)
        @test minimum(r) <= 1.05 * scaledsigmamin(rankpieces(model).J)
    end

    @testset "the same formulation whether or not the undamped solve succeeds" begin
        # A degree-13 monomial fit, whose undamped Cholesky fails or not by rounding (at these sizes, on Julia 1.13, it
        # succeeds at 400 points only; on 1.10, at 800 only): the Jacobian path every time
        for m in (200, 400, 800, 1600, 3200)
            rec = NLLSModel.recommendoptimizer(monomialmodel(m, 13); numthreads=static(1))
            @test rec.formulation === :jacobian && rec.rules.formulation === :F_illconditioned
            @test rec.features.rank_damping == (rec.features.status == DampedLinearSolve.Success ? 0.0 : first(NLLSModel.REC_RANK_DAMPINGS))
        end
        # Both outcomes forced on one model: the test run undamped, and as if the undamped solve had failed
        pc = rankpieces(monomialmodel(400, 13))
        DampedLinearSolve.solve!(pc.ls, pc.linsolver) == DampedLinearSolve.Success || DampedLinearSolve.dampen!(pc.ls, 1e-14, pc.d)
        rho, w = NLLSModel.scaled_inverse_iteration(pc.ls, pc.linsolver, sqrt.(pc.d))
        kw = (unfixed=nothing, numthreads=static(1), max_dense_bytes=NLLSModel.MAX_DENSE_BYTES)
        order = NLLSModel.varorder(pc.fdf!, monomialmodel(400, 13))
        model = monomialmodel(400, 13)
        undamped = NLLSModel.jacobian_rank_test(model, pc.ls, pc.linsolver, pc.d, rho, w, order; kw...)
        failed = NLLSModel.jacobian_rank_test(model, pc.ls, pc.linsolver, pc.d, Inf, nothing, order; kw...)
        @test undamped[1:2] == (:ran, 0.0) && failed[1:2] == (:ran, first(NLLSModel.REC_RANK_DAMPINGS))
        F(status, condition, t, λ) = NLLSModel.rule_formulation(recfeatures(; status, condition, rank_t=t, rank_damping=λ))
        @test F(DampedLinearSolve.Success, 1e15, undamped[3], 0.0) == F(DampedLinearSolve.NotPositiveDefinite, Inf, failed[3], 1e-14) == (:jacobian, :F_illconditioned)
    end

    @testset "the rank test's order: $name" for (name, make, kw) in (
            ("blocks sorted by size", () -> mixedproblem()[1], (;)),
            ("blocks sorted, an unfixed subset", () -> mixedproblem()[1], (unfixed=[1, 2, 4, 5, 7],)),
            ("eliminated, dense", () -> posechain(8, 12), (;)),
            ("eliminated, sparse", () -> posechain(8, 12), (storage=:sparse,)),
            ("eliminated, an unfixed subset", () -> posechain(8, 12), (unfixed=[2:8; 10:20],)),
            ("one variable", () -> linearmodel(scaledjacobian(3, 0.5)), (;)),
            ("a robust kernel, first order", () -> curvefit(; huber=true, outliers=5), (;)),
            ("a robust kernel, second order", gemanmcclurecurve, (;)))
        model = make()
        pc = rankpieces(model; kw...)
        lhs, rhs = permutedproducts(pc)
        @test lhs ≈ rhs rtol = 1e-13
        # The products check the order: the identity in place of a permutation that is not one does not agree
        if pc.p != 1:length(pc.p)
            lhs1, rhs1 = permutedproducts(pc, collect(1:length(pc.p)))
            @test !isapprox(lhs1, rhs1; rtol=1e-6)
        end
    end

    @testset "a robust kernel: the J of the Jacobian path" begin
        # The reweighted J the Jacobian path factorizes, not the residual's own: the plain model's J differs
        plain = rankpieces(curvefit(; outliers=5)).J
        @test !isapprox(rankpieces(curvefit(; huber=true, outliers=5)).J, plain; rtol=1e-3)
        @test !isapprox(rankpieces(gemanmcclurecurve()).J, plain; rtol=1e-3)
    end

    @testset "the explicit linear solver gives the step of :auto" begin
        # Every kind of system, residual costs only
        @testset "Hessian path: $(spec.name)" for spec in REC_MODELS
            (spec.formulation === :hessian && spec.linsolver !== :ldl || spec.kind === :sparse) || continue
            spec.name == "an unfixed adaptive kernel" && continue
            model = spec.make()
            linsolver = NLLSModel.rule_linsolver(true, spec.kind)[1]
            s1, x1 = hessianstep(model, linsolver)
            s2, x2 = hessianstep(model, :auto)
            @test s1 == s2 == DampedLinearSolve.Success
            @test x1 == x2
        end
        @testset "Jacobian path: $name" for (name, make, kw) in (
                ("dense", () -> linearmodel(scaledjacobian(3, 1 - 1e-13)), (;)),
                ("sparse", () -> weakchain(60, 1e-14), (;)),
                ("eliminated, dense", () -> posechain(8, 12), (storage=:dense,)),
                ("eliminated, sparse", () -> posechain(8, 12), (storage=:sparse,)))
            model = make()
            s1, x1 = jacobianstep(model, :qr; kw...)
            s2, x2 = jacobianstep(model, :auto; kw...)
            @test s1 == s2 == DampedLinearSolve.Success
            @test x1 == x2
        end
    end

    @testset "reduction and storage read from each prototype" begin
        model = posechain(8, 12)
        H(; kw...) = NLLSModel.hessiansystemkind(optimvars(model; numthreads=static(1), kw...)[4])
        J(; kw...) = NLLSModel.jacobianreductionstorage(optimvars_nls(model; numthreads=static(1), kw...)[5])
        for reduction in (:none, :eliminate), storage in (:dense, :sparse)
            @test NLLSModel.reductionstorage(H(; reduction, storage)) == (reduction, storage)
            @test J(; reduction, storage) == (reduction, storage)
        end
        single, _ = singleproblem(Vector{Float64}, 3, 1)
        @test NLLSModel.hessiansystemkind(optimvars(single)[4]) === :dense
        @test NLLSModel.reductionstorage(:static) == (:none, :dense)
    end

    @testset "conditioning" begin
        # Linear least squares with a known scaled condition number: the estimate never overstates it, and is within the
        # largest eigenvalue of the scaled matrix, 1 + (n - 1) c, of it (measured: at most 6.006, for n = 6)
        @testset "n = $n, c = $c" for n in (2, 3, 6), c in (0.0, 0.5, 0.9, 0.999, 1 - 1e-6, 1 - 1e-10, 1 - 1e-13)
            s = [10.0^(k % 4 - 2) for k in 1:n] # Badly scaled, which the estimate does not see
            rec = NLLSModel.recommendoptimizer(linearmodel(scaledjacobian(n, c, s)); numthreads=static(1))
            truth = scaledcondition(n, c)
            @test rec.features.condition <= truth * (1 + 1e-6)
            @test truth / rec.features.condition <= 1.001 * (1 + (n - 1) * c) + 0.01
            @test truth / rec.features.condition <= 6.01
            @test rec.formulation === (c == 1 - 1e-13 ? :jacobian : :hessian)
        end
        # Badly scaled but well conditioned: a small condition estimate and a large spread
        rec = NLLSModel.recommendoptimizer(linearmodel(scaledjacobian(3, 0.5, [1.0, 1e4, 1e-3])); numthreads=static(1))
        @test rec.features.condition < 4
        @test rec.features.diagspread ≈ 1e14
        @test rec.formulation === :hessian && rec.iterator === :levenbergmarquardt_dtd
        # Rank-deficient (exactly: x₁ - x₂ alone): the factorization fails, so Inf; the damped one does, and the rank test
        # finds J rank-deficient, so the normal equations
        rec = NLLSModel.recommendoptimizer(linearmodel([1.0 -1.0]); numthreads=static(1))
        @test rec.features.status == DampedLinearSolve.NotPositiveDefinite
        @test rec.features.condition == Inf
        @test rec.formulation === :hessian && rec.rules.formulation === :F_rankdeficient
        # A zero diagonal entry (a variable no cost moves): Inf, and an infinite spread
        rec = NLLSModel.recommendoptimizer(linearmodel([1.0 0.0; 2.0 0.0]); numthreads=static(1))
        @test rec.features.diag_min == 0 && rec.features.diagspread == Inf && rec.features.condition == Inf
        @test rec.formulation === :hessian
        # Finitely and badly conditioned: the Jacobian path, with QR
        rec = NLLSModel.recommendoptimizer(linearmodel(scaledjacobian(3, 1 - 1e-13)); numthreads=static(1))
        @test isfinite(rec.features.condition) && rec.features.condition >= NLLSModel.REC_JACOBIAN_CONDITION
        @test rec.formulation === :jacobian && rec.linsolver === :qr
        # The estimate of a system that does not factorize
        ls = DampedLinearSolve.makehessiansystem(zeros(2, 2), :cholesky)
        @test NLLSModel.scaled_inverse_iteration(ls, :cholesky, ones(2))[1] == Inf
        # The start of inverse iteration is a fixed unit vector
        @test norm(NLLSModel.inverse_iteration_start(5)) ≈ 1
        @test NLLSModel.inverse_iteration_start(5) == NLLSModel.inverse_iteration_start(5)
    end

    @testset "not finite at the current values" begin
        err = try NLLSModel.recommendoptimizer(linearmodel(Matrix(1.0I, 2, 2), zeros(2), [NaN, 1.0])); nothing catch e e end
        @test err isa ArgumentError && occursin("cannot be linearized at its current values", err.msg)
        @test occursin("NaN", err.msg)
        # A finite cost with a non-finite Hessian: (floatmax x₂)² overflows, but x₂ = 0
        model = Model(Vector{Float64}, LinearResidualDynamic)
        addvariable!(model, [1.0, 0.0])
        addcost!(model, LinearResidualDynamic(zeros(2), [1.0 0.0; 0.0 floatmax()], 1))
        err = try NLLSModel.recommendoptimizer(model); nothing catch e e end
        @test err isa ArgumentError && occursin("not finite", err.msg)
    end

    @testset "show: the one-line form" begin
        rec = recby(formulation=:hessian, reduction=:eliminate, storage=:sparse, linsolver=:cholesky, iterator=:dogleg,
            features=recfeatures(hessiansystem=:schur_sparse, retained=6, eliminated=9, dimension=15))
        @test sprint(show, rec) == "OptimizationRecommendation(formulation = :hessian, reduction = :eliminate, storage = :sparse, linsolver = :cholesky, iterator = :dogleg)"
        @test repr(rec) == sprint(show, rec)
    end

    @testset "show: the long form, $(spec.name)" for spec in REC_MODELS
        rec = NLLSModel.recommendoptimizer(spec.make(); numthreads=static(1))
        text = longform(rec)
        for name in (:formulation, :reduction, :storage, :linsolver, :iterator)
            @test occursin("$(rpad(name, 11)) = $(repr(getfield(rec, name)))", text)
        end
        for reason in (NLLSModel.formulation_reason, NLLSModel.reduction_reason, NLLSModel.storage_reason,
                NLLSModel.linsolver_reason, NLLSModel.iterator_reason)
            @test occursin(flat(reason(rec)), flat(text))
        end
        @test occursin("a cost evaluation", text) && occursin("an undamped solve", text)
        @test occursin("The thresholds are starting values", flat(text))
        @test all(line -> textwidth(line) <= 80, split(text, '\n'))
        @test !occursin(":auto)", text) && !occursin("= :auto,", text) # Every choice is explicit
        # The call: exactly the expected arguments
        positional, keywords, optkw = printedcall(rec)
        @test positional[1] === :model
        @test optkw == Dict(:iterator => QuoteNode(rec.iterator), :linsolver => QuoteNode(rec.linsolver))
        expected = Dict{Symbol, Any}(:formulation => QuoteNode(rec.formulation), :reduction => QuoteNode(rec.reduction),
            :storage => QuoteNode(rec.storage))
        Threads.nthreads() == 1 || (expected[:numthreads] = 1)
        @test keywords == expected
    end

    @testset "show: the call, with unfixed = $(repr(unfixed))" for (unfixed, printed) in (
            ([2:8; 10:20], :unfixed), (3, 3), (ChainLandmark{Float64}, :(ChainLandmark{Float64})))
        model = posechain(8, 12)
        rec = NLLSModel.recommendoptimizer(model; unfixed, numthreads=Threads.nthreads(), max_dense_bytes=10^6)
        _, keywords, _ = printedcall(rec)
        @test keywords == Dict(:formulation => QuoteNode(rec.formulation), :reduction => QuoteNode(rec.reduction),
            :storage => QuoteNode(rec.storage), :unfixed => printed, :max_dense_bytes => 10^6)
        @test occursin("`unfixed` stands for", longform(rec)) == (printed === :unfixed)
        # numthreads only when it is not the default
        rec = NLLSModel.recommendoptimizer(model; unfixed, numthreads=Threads.nthreads() + 1)
        _, keywords, _ = printedcall(rec)
        @test keywords[:numthreads] == Threads.nthreads() + 1
        @test !haskey(keywords, :max_dense_bytes)
    end

    @testset "show: every reason" begin
        # Each rule's sentence, with the feature values that decided it
        reason(f; kw...) = flat(f(recby(; kw...)))
        @test occursin("not residuals", reason(NLLSModel.formulation_reason; features=recfeatures(allresiduals=false)))
        @test occursin("a diagonal entry that is not positive (the smallest is -2.0; the undamped solve returned NotPositiveDefinite)",
            reason(NLLSModel.formulation_reason; features=recfeatures(status=DampedLinearSolve.NotPositiveDefinite, condition=Inf, diag_min=-2.0)))
        @test occursin("could not be factorized (Indefinite), undamped or with any scaled damping up to 1.0e-10",
            reason(NLLSModel.formulation_reason; features=recfeatures(status=DampedLinearSolve.Indefinite, condition=Inf, ranktest=:nofactorization)))
        deficient = recfeatures(condition=3e16, rank_t=2e-16)
        @test occursin("Hessian, 3.0e16, is at least 1.0e12, so the rank of the Jacobian decides: along the Hessian's " *
            "weakest direction the column-scaled Jacobian has norm 2.0e-16, below 1.0e-12, so J is numerically rank-deficient",
            reason(NLLSModel.formulation_reason; features=deficient))
        @test occursin("the undamped Hessian could not be factorized (NotPositiveDefinite) (the test used a scaled damping of 1.0e-14)",
            reason(NLLSModel.formulation_reason; features=recfeatures(condition=Inf, status=DampedLinearSolve.NotPositiveDefinite, rank_t=0.0)))
        @test occursin("has norm 2.0e-8, at least 1.0e-12, so J has full numerical rank, and the condition number of the " *
            "column-scaled J is at least 5.0e7", reason(NLLSModel.formulation_reason; formulation=:jacobian, features=recfeatures(condition=1e15, rank_t=2e-8)))
        @test occursin("could not be factorized (NotPositiveDefinite) (the test used a scaled damping of 1.0e-14), so the rank of the Jacobian decides: along",
            reason(NLLSModel.formulation_reason; formulation=:jacobian, features=recfeatures(condition=Inf, status=DampedLinearSolve.NotPositiveDefinite, rank_t=2e-8)))
        @test occursin("the undamped Hessian could not be factorized (NotPositiveDefinite), so the rank of the Jacobian would decide, but the Jacobian path cannot be built for this model (why not)",
            reason(NLLSModel.formulation_reason; features=recfeatures(condition=Inf, status=DampedLinearSolve.NotPositiveDefinite, ranktest=:nojacobian), fallback="why not"))
        @test occursin("12.3, is below 1.0e12", reason(NLLSModel.formulation_reason; features=recfeatures(condition=12.345)))
        @test occursin("3.0e13, is at least 1.0e12", reason(NLLSModel.formulation_reason; formulation=:jacobian, features=recfeatures(condition=3e13)))
        @test occursin("cannot be built for this model (why not)", reason(NLLSModel.formulation_reason;
            features=recfeatures(condition=3e13), rule_f=:F_jacobian_unavailable, fallback="why not"))
        schur = recfeatures(hessiansystem=:schur_dense, retained=6, eliminated=9, dimension=15)
        @test occursin("9 of the 15 dimensions are eliminated", reason(NLLSModel.reduction_reason; reduction=:eliminate, features=schur))
        @test occursin("eliminated by QR", reason(NLLSModel.reduction_reason; formulation=:jacobian, reduction=:eliminate, features=schur))
        @test occursin("single unfixed variable", reason(NLLSModel.reduction_reason; features=recfeatures(nvariables=1)))
        @test occursin("Jacobian path: no variables", reason(NLLSModel.reduction_reason; formulation=:jacobian))
        @test occursin("no eliminable variables", reason(NLLSModel.reduction_reason))
        @test occursin("the 6×6 reduced system is stored sparse, from the predicted fill",
            reason(NLLSModel.storage_reason; reduction=:eliminate, storage=:sparse, features=schur))
        @test occursin("static (stack-allocated) 2×2", reason(NLLSModel.storage_reason; features=recfeatures(hessiansystem=:static, dimension=2)))
        @test occursin("the 4×4 Hessian is stored dense, from its sparsity", reason(NLLSModel.storage_reason))
        @test occursin("a single variable's always is", reason(NLLSModel.storage_reason; features=recfeatures(nvariables=1)))
        @test occursin("the 6×4 Jacobian is stored sparse", reason(NLLSModel.storage_reason; formulation=:jacobian, storage=:sparse))
        @test occursin("Jacobian path: the reduced system is stored dense", reason(NLLSModel.storage_reason; formulation=:jacobian, reduction=:eliminate))
        @test occursin("QR", reason(NLLSModel.linsolver_reason; formulation=:jacobian))
        @test occursin("can be indefinite", reason(NLLSModel.linsolver_reason; features=recfeatures(allresiduals=false)))
        @test occursin("sparse Hessian", reason(NLLSModel.linsolver_reason; features=recfeatures(hessiansystem=:sparse)))
        @test occursin("Schur complement", reason(NLLSModel.linsolver_reason; features=schur))
        @test occursin("static Hessian", reason(NLLSModel.linsolver_reason; features=recfeatures(hessiansystem=:static)))
        @test occursin("dense Hessian", reason(NLLSModel.linsolver_reason))
        @test occursin("dogleg, which does not solve again", reason(NLLSModel.iterator_reason))
        @test occursin("spread (largest / smallest), 1.0e7, is at least 1.0e6", reason(NLLSModel.iterator_reason; features=recfeatures(diagspread=1e7)))
        lm = reason(NLLSModel.iterator_reason; features=recfeatures(status=DampedLinearSolve.Indefinite, condition=Inf, t_solve=1e-6))
        @test occursin("dogleg is not chosen because the undamped solve returned Indefinite", lm)
        @test occursin("Inf, is not below 1.0e8", lm)
        @test occursin("a solve (1.0 μs) costs less than a linearization (10.0 μs)", lm)
        # Times in the unit that reads best
        @test NLLSModel.fmttime(2.5e-8) == "25.0 ns"
        @test NLLSModel.fmttime(2.5e-5) == "25.0 μs"
        @test NLLSModel.fmttime(2.5e-2) == "25.0 ms"
        @test NLLSModel.fmttime(25.0) == "25.0 s"
        @test NLLSModel.fmtnum(Inf) == "Inf" && NLLSModel.fmtnum(123456) == "123000.0"
        # Wrapping: at most the width, indented, every word kept
        text = sprint(NLLSModel.printwrapped, repeat("word ", 40), 4)
        @test all(line -> textwidth(line) <= 80 && startswith(line, "    "), split(chomp(text), '\n'))
        @test split(text) == fill("word", 40)
    end

    @testset "keywords are passed on" begin
        model = posechain(8, 12)
        rec = NLLSModel.recommendoptimizer(model; unfixed=[2:8; 10:20], numthreads=1, max_dense_bytes=10^6)
        @test rec.unfixed == [2:8; 10:20] && rec.numthreads === 1 && rec.max_dense_bytes == 10^6
        @test rec.features.nvariables == 18
        rec = NLLSModel.recommendoptimizer(model; unfixed=1)
        @test rec.features.nvariables == 1 && rec.features.dimension == 3 && rec.reduction === :none
        # max_dense_bytes decides the storage as it does for optimvars
        small, _ = chainproblem(5, 2, 9)
        @test NLLSModel.recommendoptimizer(small; max_dense_bytes=799).storage === :sparse
        @test NLLSModel.recommendoptimizer(small; max_dense_bytes=800).storage === :dense
    end
end
