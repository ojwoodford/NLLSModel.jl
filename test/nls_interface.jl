using Test
using NLLSModel
using NLPModels
using Static, StaticArrays, SparseArrays, LinearAlgebra
import OptimizationVariableInterface: nvars, update, update!, syncvars!, ZeroToInfScalar

include("fixtures/loader.jl")
fixture("nls_interface")

# ---------------------------------------------------------------------------
# Test Residual Definitions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

@testset "NLS Interface - Core Functors & Traits" begin

    @testset "6.1.1 Finite-Difference Jacobian Validation" begin
        # 1. Static SVector variable
        prob = Model(SVector{2, Float64}, Union{TestRosenbrock1, TestRosenbrock2})
        addvariable!(prob, SVector(-1.2, 1.0))
        addcost!(prob, TestRosenbrock1())
        addcost!(prob, TestRosenbrock2())

        flat = flat_optimvars_nls(prob)
        r = zeros(flat.layout_jacobian.M)
        J = copy(flat.J_proto)
        c = flat.resjac!(r, J, flat.x0)
        J_fd = finite_diff_jac(flat.res!, flat.x0, flat.layout_jacobian.M)
        @test isapprox(J, J_fd, atol=1e-5)
        @test isapprox(c, 0.5 * dot(r, r); atol=1e-12)
        @test isapprox(c, flat.f(flat.x0); atol=1e-12)
        @test isapprox(c, cost(prob.variables, prob.costs); atol=1e-12)

        # 2. Dynamic Vector{Float64} variable
        prob_dyn = Model(Vector{Float64}, DynamicRes)
        addvariable!(prob_dyn, [1.0, 2.0, 3.0])
        addcost!(prob_dyn, DynamicRes(3))
        flat_dyn = flat_optimvars_nls(prob_dyn)
        r_dyn = zeros(3)
        J_dyn = copy(flat_dyn.J_proto)
        flat_dyn.resjac!(r_dyn, J_dyn, flat_dyn.x0)
        J_dyn_fd = finite_diff_jac(flat_dyn.res!, flat_dyn.x0, 3)
        @test isapprox(J_dyn, J_dyn_fd, atol=1e-5)

        # 3. Multi-dimensional residual (nres = 2)
        prob_m = Model(SVector{2, Float64}, MultiDimRes)
        addvariable!(prob_m, SVector(3.0, 5.0))
        addcost!(prob_m, MultiDimRes())
        flat_m = flat_optimvars_nls(prob_m)
        r_m = zeros(2)
        J_m = copy(flat_m.J_proto)
        flat_m.resjac!(r_m, J_m, flat_m.x0)
        J_m_fd = finite_diff_jac(flat_m.res!, flat_m.x0, 2)
        @test isapprox(J_m, J_m_fd, atol=1e-5)

        # 4. Scalar varindices (ndeps = 1, Int)
        prob_s = Model(Float64, ScalarVarIndRes)
        addvariable!(prob_s, 5.0)
        addcost!(prob_s, ScalarVarIndRes())
        flat_s = flat_optimvars_nls(prob_s)
        r_s = zeros(1)
        J_s = copy(flat_s.J_proto)
        flat_s.resjac!(r_s, J_s, flat_s.x0)
        J_s_fd = finite_diff_jac(flat_s.res!, flat_s.x0, 1)
        @test isapprox(J_s, J_s_fd, atol=1e-5)

        # 5. Scaled linear kernel
        prob_sc = Model(Float64, RobustWrapper{ScalarVarIndRes, Scaled{Float64, NoRobust}})
        addvariable!(prob_sc, 5.0)
        addcost!(prob_sc, RobustWrapper(ScalarVarIndRes(), Scaled(NoRobust(), 2.5)))
        flat_sc = flat_optimvars_nls(prob_sc)
        r_sc = zeros(1)
        J_sc = copy(flat_sc.J_proto)
        c_sc = flat_sc.resjac!(r_sc, J_sc, flat_sc.x0)
        J_sc_fd = finite_diff_jac(flat_sc.res!, flat_sc.x0, 1)
        @test isapprox(J_sc, J_sc_fd, atol=1e-5)
        @test isapprox(c_sc, 0.5 * dot(r_sc, r_sc); atol=1e-12)
    end

    @testset "6.1.2 Normal-Equation Equivalence" begin
        # NoRobust
        prob1 = Model(SVector{2, Float64}, Union{TestRosenbrock1, TestRosenbrock2})
        addvariable!(prob1, SVector(-1.2, 1.0))
        addcost!(prob1, TestRosenbrock1())
        addcost!(prob1, TestRosenbrock2())

        f_norm, fdf!_norm, x0_norm, _ = optimvars(prob1; numthreads=static(1))
        g_norm = zeros(2)
        H_norm = zeros(2, 2)
        c_norm = fdf!_norm(g_norm, H_norm, x0_norm)

        f_jac, res!_jac, resjac!_jac, x0_jac, J_proto = optimvars_nls(prob1, numthreads=static(1))
        r_jac = zeros(2)
        J_jac = copy(J_proto)
        c_jac = resjac!_jac(r_jac, J_jac, x0_jac)

        @test isapprox(J_jac' * r_jac, g_norm; atol=1e-12)
        @test isapprox(J_jac' * J_jac, H_norm; atol=1e-12)
        @test isapprox(c_jac, c_norm; atol=1e-12)

        # A first-order kernel (zero second derivative): IRLS
        prob2 = Model(Float64, RobustWrapper{ScalarVarIndRes, HuberKernel{Float64, Static.False}})
        addvariable!(prob2, 5.0)
        addcost!(prob2, RobustWrapper(ScalarVarIndRes(), HuberKernel(1.0)))
        _, fdf2, x02, _ = optimvars(prob2; numthreads=static(1))
        g2 = zeros(1)
        H2 = zeros(1, 1)
        c2 = fdf2(g2, H2, x02)

        _, _, resjac2, _, J_proto2 = optimvars_nls(prob2, numthreads=static(1))
        r2 = zeros(1)
        J2 = copy(J_proto2)
        c_jac2 = resjac2(r2, J2, x02)
        @test isapprox(J2' * r2, g2; atol=1e-12)
        @test isapprox(c_jac2, c2; atol=1e-12)

        # A second-order kernel: IRLS with the Triggs correction, across scales
        for scale in [1e-3, 1.0, 1e3]
            prob3 = Model(Float64, RobustWrapper{ScalarVarIndRes, HuberKernel{Float64, Static.True}})
            addvariable!(prob3, 5.0 * scale)
            addcost!(prob3, RobustWrapper(ScalarVarIndRes(), Huber2oKernel(2.0 * scale)))
            _, fdf3, x03, _ = optimvars(prob3; numthreads=static(1))
            g3 = zeros(1)
            H3 = zeros(1, 1)
            c3 = fdf3(g3, H3, x03)

            _, _, resjac3, _, J_proto3 = optimvars_nls(prob3, numthreads=static(1))
            r3 = zeros(1)
            J3 = copy(J_proto3)
            c_jac3 = resjac3(r3, J3, x03)

            @test isapprox(J3' * r3, g3; atol=1e-9 * scale)
            @test isapprox(J3' * J3, H3; atol=1e-9 * scale)
            @test isapprox(c_jac3, c3; atol=1e-9 * scale^2)
        end
    end

    @testset "6.1.3 Reused Sparse J Zeroing & Cost Preservation" begin
        prob_zw = Model(Float64, RobustWrapper{ScalarVarIndRes, ZeroWeightKernel})
        addvariable!(prob_zw, 10.0)
        addcost!(prob_zw, RobustWrapper(ScalarVarIndRes(), ZeroWeightKernel()))

        flat = flat_optimvars_nls(prob_zw, storage=:sparse)
        J_reused = copy(flat.J_proto)
        fill!(J_reused.nzval, 999.0) # Pre-fill with non-zero garbage
        r = zeros(1)

        c = flat.resjac!(r, J_reused, flat.x0)
        @test r[1] == 0.0
        @test all(J_reused.nzval .== 0.0)
        @test c == 0.5 * 15.0 # Cost is preserved despite zero rows!
    end

    @testset "6.1.4 Non-Finite Trial-Step Propagation" begin
        for mode in [:nan, :inf, :overflow]
            prob = Model(SVector{1, Float64}, NonFiniteRes)
            addvariable!(prob, SVector(1.0))
            addcost!(prob, NonFiniteRes(mode))

            flat = flat_optimvars_nls(prob)
            r = zeros(1)
            J = copy(flat.J_proto)

            # res! and resjac! must not throw exceptions
            c_res = flat.res!(r, flat.x0)
            @test !isfinite(c_res)
            @test any(!isfinite, r)

            c_resjac = flat.resjac!(r, J, flat.x0)
            @test !isfinite(c_resjac)
            @test any(!isfinite, r)
            @test any(!isfinite, J)
        end
    end

    @testset "6.1.5 Lazy kernel_dc_tol, Clamping & DomainError" begin
        # 1. Clamped case: slightly negative derivative within tolerance
        prob_clamp = Model(Float64, RobustWrapper{ScalarVarIndRes, MockClampingKernel})
        addvariable!(prob_clamp, 3.0)
        addcost!(prob_clamp, RobustWrapper(ScalarVarIndRes(), MockClampingKernel(-1e-15)))

        flat_clamp = flat_optimvars_nls(prob_clamp)
        r = zeros(1)
        J = copy(flat_clamp.J_proto)

        @test NLLSModel.clamped_count(flat_clamp) == 0
        c = flat_clamp.resjac!(r, J, flat_clamp.x0)
        @test NLLSModel.clamped_count(flat_clamp) == 1
        @test r[1] == 0.0
        @test J[1, 1] == 0.0

        NLLSModel.reset_clamped_count!(flat_clamp)
        @test NLLSModel.clamped_count(flat_clamp) == 0

        # 2. Domain error: negative derivative beyond tolerance
        prob_domain = Model(Float64, RobustWrapper{ScalarVarIndRes, MockClampingKernel})
        addvariable!(prob_domain, 3.0)
        addcost!(prob_domain, RobustWrapper(ScalarVarIndRes(), MockClampingKernel(-0.1)))

        flat_domain = flat_optimvars_nls(prob_domain)
        r_dom = zeros(1)
        J_dom = copy(flat_domain.J_proto)
        @test_throws DomainError flat_domain.resjac!(r_dom, J_dom, flat_domain.x0)
    end

    @testset "6.1.7 Fixed AbstractAdaptiveResidual Support" begin
        prob = Model(Union{ContaminatedGaussian{Float64, 1}, Float64}, TestAdaptiveRes)
        kernel = ContaminatedGaussian(1.0, 5.0, 0.8, static(1))
        addvariable!(prob, kernel) # var 1 = adaptive kernel
        addvariable!(prob, 2.0)    # var 2 = value

        addcost!(prob, TestAdaptiveRes(SVector(1, 2)))

        # Unfixed adaptive variable must throw ArgumentError
        @test_throws ArgumentError optimvars_nls(prob; unfixed=nothing)

        # Fixed adaptive variable (unfixed = 2) must succeed
        flat_adapt = flat_optimvars_nls(prob; unfixed=2)
        r = zeros(1)
        J = copy(flat_adapt.J_proto)
        c = flat_adapt.resjac!(r, J, flat_adapt.x0)
        @test isfinite(c)
        @test isfinite(r[1])
        @test isfinite(J[1, 1])
    end

    @testset "6.1.8 Guards & Trait Checks" begin
        # 1. There is no robust mode; the linear-only paths throw for a non-linear kernel, and islinear says why
        prob_huber = Model(Float64, RobustWrapper{ScalarVarIndRes, HuberKernel{Float64, Static.False}})
        addvariable!(prob_huber, 3.0)
        addcost!(prob_huber, RobustWrapper(ScalarVarIndRes(), HuberKernel(1.0)))
        @test_throws MethodError optimvars_nls(prob_huber; robust=:irls)
        @test_throws MethodError flat_optimvars_nls(prob_huber; robust=:none)
        @test !islinear(optimvars_nls(prob_huber)[3])
        @test !islinear(flat_optimvars_nls(prob_huber))
        @test_throws ArgumentError NLLSModel.nls_callbacks(prob_huber)
        @test_throws ArgumentError NLLSNLSModel(prob_huber)
        # islinear: all linear, mixed, all robust
        prob_lin = Model(Float64, RobustWrapper{ScalarVarIndRes, Scaled{Float64, NoRobust}})
        addvariable!(prob_lin, 3.0)
        addcost!(prob_lin, RobustWrapper(ScalarVarIndRes(), Scaled(NoRobust(), 2.0)))
        @test islinear(optimvars_nls(prob_lin)[3]) && islinear(optimvars_nls(prob_lin)[2]) && islinear(flat_optimvars_nls(prob_lin))
        prob_mix = Model(Float64, Union{RobustWrapper{ScalarVarIndRes, Scaled{Float64, NoRobust}}, RobustWrapper{ScalarVarIndRes, HuberKernel{Float64, Static.False}}})
        addvariable!(prob_mix, 3.0)
        addcost!(prob_mix, RobustWrapper(ScalarVarIndRes(), Scaled(NoRobust(), 2.0)))
        addcost!(prob_mix, RobustWrapper(ScalarVarIndRes(), HuberKernel(1.0)))
        @test !islinear(optimvars_nls(prob_mix)[3])
        @test_throws ArgumentError NLLSModel.nls_callbacks(prob_mix)
        @test_throws ArgumentError NLLSNLSModel(prob_mix)
        @test NLLSModel.nls_callbacks(prob_lin) isa NamedTuple
        @test NLLSNLSModel(prob_lin) isa AbstractNLSModel

        # 2. Scaled(NoRobust(), -1.0) throws
        prob_neg = Model(Float64, RobustWrapper{ScalarVarIndRes, Scaled{Float64, NoRobust}})
        addvariable!(prob_neg, 3.0)
        addcost!(prob_neg, RobustWrapper(ScalarVarIndRes(), Scaled(NoRobust(), -1.0)))
        @test_throws ArgumentError NLLSModel.nls_callbacks(prob_neg)

        # 3. Variables: the numerical check decides, nothing is declared
        @test !isdefined(NLLSModel, :isadditive)
        for v in (2.0, SVector(1.0, 2.0, 3.0), [1.0, 2.0], EuclidVec([1.0, 2.0, 3.0]))
            @test NLLSModel.check_additive_variable(v)
        end
        argmsg(v) = try NLLSModel.check_additive_variable(v); "" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError" end
        @test occursin("plain addition", argmsg(WrappedAngle(3.1))) && occursin("WrappedAngle", argmsg(WrappedAngle(3.1)))
        # ZeroToInfScalar is neither a number nor an AbstractVector, so it cannot be packed
        @test occursin("cannot be packed", argmsg(ZeroToInfScalar(1.0))) && occursin("ZeroToInfScalar", argmsg(ZeroToInfScalar(1.0)))
        @test occursin("cannot be packed", argmsg(UnbuildableVec([1.0, 2.0], 0))) # An AbstractVector with no constructor from its values
        @test occursin("cannot be packed", argmsg(:symbol))
        # A vector longer than its nvars (ambient coordinates) is rejected before it is packed (AN4)
        lenmsg = "length(v) == nvars(v)"
        @test occursin(lenmsg, argmsg(AmbientVec([1.0, 0.0, 0.0, 0.0]))) && occursin("AmbientVec", argmsg(AmbientVec([1.0, 0.0, 0.0, 0.0])))
        prob_amb = Model(AmbientVec, EuclidRes)
        addvariable!(prob_amb, AmbientVec([1.0, 0.0, 0.0, 0.0]))
        addcost!(prob_amb, EuclidRes(1))
        @test_throws lenmsg flat_optimvars_nls(prob_amb)
        @test_throws lenmsg NLLSNLSModel(prob_amb)
        # Through flat_optimvars_nls, and once per variable type: 1 000 variables of one type, one check
        prob_rot = Model(WrappedAngle, AngleRes)
        addvariable!(prob_rot, WrappedAngle(3.1))
        addcost!(prob_rot, AngleRes(1))
        @test_throws ArgumentError flat_optimvars_nls(prob_rot)
        prob_many = Model(Union{EuclidVec, Float64}, EuclidRes)
        for k in 1:1000
            addvariable!(prob_many, EuclidVec([0.0, Float64(k)]))
            addcost!(prob_many, EuclidRes(k))
        end
        addvariable!(prob_many, 1.0) # A second type, checked once too
        EUCLID_UPDATES[] = 0
        flat_many = flat_optimvars_nls(prob_many)
        @test EUCLID_UPDATES[] == 1
        @test length(flat_many.x0) == 2001
        # Vectors of different lengths: one check for the type
        prob_vec = Model(Vector{Float64}, EuclidRes)
        addvariable!(prob_vec, [1.0])
        addvariable!(prob_vec, [1.0, 2.0, 3.0])
        @test length(flat_optimvars_nls(prob_vec).x0) == 4

        # 4. Dense J with NaNs has off-structure entries zeroed
        prob_dense = Model(SVector{2, Float64}, TestRosenbrock1)
        addvariable!(prob_dense, SVector(1.0, 2.0))
        addcost!(prob_dense, TestRosenbrock1())
        _, _, resjac_d, x0_d, _ = optimvars_nls(prob_dense, storage=:dense)
        J_nan = fill(NaN, 1, 2)
        r_d = zeros(1)
        resjac_d(r_d, J_nan, x0_d)
        @test all(isfinite, J_nan)
    end
end

@testset "NLS Interface - objgrad! evaluates each residual once (AN3)" begin
    prob = Model(SVector{2, Float64}, Union{CountedNLS{TestRosenbrock1}, CountedNLS{TestRosenbrock2}})
    addvariable!(prob, SVector(-1.2, 1.0))
    addcost!(prob, CountedNLS(TestRosenbrock1()))
    addcost!(prob, CountedNLS(TestRosenbrock2()))
    ncosts = 2
    nls = NLLSNLSModel(prob; numthreads=static(1))
    evals(fun) = (NLS_EVALS[] = 0; fun(); NLS_EVALS[])
    g, g2 = zeros(2), zeros(2)
    # grad! at a new point: one pass over the costs
    @test evals(() -> grad!(nls, [0.3, 0.4], g)) == ncosts
    # objgrad! at a new point: one pass too
    @test evals(() -> objgrad!(nls, [0.5, 0.6], g2)) == ncosts
    # The same cost and gradient as obj and grad! on a fresh model
    fresh = NLLSNLSModel(prob; numthreads=static(1))
    x = [0.7, -0.2]
    c = obj(fresh, x)
    @test grad!(fresh, x, g) == g
    @test objgrad!(nls, x, g2) == (c, g)
    # obj, then objgrad! at the same point: one pass, and the same values
    x = [-0.1, 0.9]
    c = obj(nls, x)
    @test evals(() -> objgrad!(nls, x, g2)) == ncosts
    @test objgrad!(nls, x, g2)[1] == c && g2 == grad!(fresh, x, g)
end

