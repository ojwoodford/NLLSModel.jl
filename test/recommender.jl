using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian
import DampedLinearSolve

include("fixtures/loader.jl")
fixture("recommender")

# RC3: the recommender's features, rules and choices (src/recommender.jl)

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
        @test F(condition=NLLSModel.REC_TALL_CONDITION) == (:tall, :F_illconditioned)
        @test F(condition=prevfloat(NLLSModel.REC_TALL_CONDITION)) == (:normal, :F_wellconditioned)
        @test F(condition=Inf, status=DampedLinearSolve.NotPositiveDefinite) == (:normal, :F_singular)
        @test F(condition=1e20, allresiduals=false) == (:normal, :F_nonresidual)
        @test F(condition=Inf, allresiduals=false) == (:normal, :F_nonresidual)
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
        @test NLLSModel.rule_formulation(f)[1] === (spec.name == "an unfixed adaptive kernel" ? :tall : rec.formulation)
        @test rec.iterator === NLLSModel.rule_iterator(f)[1]

        # reduction and storage are what optimvars (optimvars_nls on the Jacobian path) builds with :auto
        if rec.formulation === :normal
            @test rec.rules.reduction === rec.rules.storage === :RS_hessian
            @test (rec.reduction, rec.storage) == NLLSModel.reductionstorage(NLLSModel.hessiansystemkind(optimvars(model; numthreads=static(1))[4]))
        else
            @test rec.rules.reduction === rec.rules.storage === :RS_jacobian
            @test rec.rules.linsolver === :L_tall
            @test (rec.reduction, rec.storage) == NLLSModel.jacobianreductionstorage(optimvars_nls(model; numthreads=static(1))[5])
        end
        @test (rec.reduction === :eliminate) == (spec.kind in (:schur_dense, :schur_sparse))

        # Two calls agree on everything but the timings (and the iterator, where the timings decided it)
        rec2 = NLLSModel.recommendoptimizer(model; numthreads=static(1))
        @test untimed(rec2.features) == untimed(f)
        @test (rec2.formulation, rec2.reduction, rec2.storage, rec2.linsolver) == (rec.formulation, rec.reduction, rec.storage, rec.linsolver)
        @test rec2.fallback == rec.fallback
        timingdecided(f) || @test rec2.iterator === rec.iterator
        @test model.variables == start
    end

    @testset "the fallback from the Jacobian path" begin
        rec = NLLSModel.recommendoptimizer(adaptivemodel(); numthreads=static(1))
        @test rec.features.condition >= NLLSModel.REC_TALL_CONDITION
        @test rec.formulation === :normal && rec.rules.formulation === :F_tall_unavailable
        @test occursin("adaptive", rec.fallback)
        @test rec.rules.reduction === :RS_hessian && rec.rules.linsolver === :L_dense
    end

    @testset "the explicit linear solver gives the step of :auto" begin
        # Every kind of system, residual costs only
        @testset "Hessian path: $(spec.name)" for spec in REC_MODELS
            (spec.formulation === :normal && spec.linsolver !== :ldl || spec.kind === :sparse) || continue
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
            s1, x1 = tallstep(model, :qr; kw...)
            s2, x2 = tallstep(model, :auto; kw...)
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
            @test rec.formulation === (c == 1 - 1e-13 ? :tall : :normal)
        end
        # Badly scaled but well conditioned: a small condition estimate and a large spread
        rec = NLLSModel.recommendoptimizer(linearmodel(scaledjacobian(3, 0.5, [1.0, 1e4, 1e-3])); numthreads=static(1))
        @test rec.features.condition < 4
        @test rec.features.diagspread ≈ 1e14
        @test rec.formulation === :normal && rec.iterator === :levenbergmarquardt_dtd
        # Rank-deficient (exactly: x₁ - x₂ alone): the factorization fails, so Inf, and the normal equations
        rec = NLLSModel.recommendoptimizer(linearmodel([1.0 -1.0]); numthreads=static(1))
        @test rec.features.status == DampedLinearSolve.NotPositiveDefinite
        @test rec.features.condition == Inf
        @test rec.formulation === :normal && rec.rules.formulation === :F_singular
        # A zero diagonal entry (a variable no cost moves): Inf, and an infinite spread
        rec = NLLSModel.recommendoptimizer(linearmodel([1.0 0.0; 2.0 0.0]); numthreads=static(1))
        @test rec.features.diag_min == 0 && rec.features.diagspread == Inf && rec.features.condition == Inf
        @test rec.formulation === :normal
        # Finitely and badly conditioned: the Jacobian path, with QR
        rec = NLLSModel.recommendoptimizer(linearmodel(scaledjacobian(3, 1 - 1e-13)); numthreads=static(1))
        @test isfinite(rec.features.condition) && rec.features.condition >= NLLSModel.REC_TALL_CONDITION
        @test rec.formulation === :tall && rec.linsolver === :qr
        # The estimate of a system that does not factorize
        ls = DampedLinearSolve.makehessiansystem(zeros(2, 2), :cholesky)
        @test NLLSModel.scaled_condition_estimate(ls, :cholesky, ones(2)) == Inf
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
