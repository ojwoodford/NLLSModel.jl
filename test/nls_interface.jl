using Test
using NLLSModel
using NLPModels
using Static, StaticArrays, SparseArrays, LinearAlgebra
import OptimizationVariableInterface: nvars, update, update!, syncvars!, ZeroToInfScalar
import JSOSolvers
import LeastSquaresOptim

# ---------------------------------------------------------------------------
# Test Residual Definitions
# ---------------------------------------------------------------------------

struct TestRosenbrock1 <: AbstractResidual end
NLLSModel.ndeps(::TestRosenbrock1) = static(1)
NLLSModel.nres(::TestRosenbrock1) = static(1)
NLLSModel.varindices(::TestRosenbrock1) = SVector(1)
NLLSModel.getvars(::TestRosenbrock1, vars::AbstractVector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::TestRosenbrock1, x) = SVector(1.0 - x[1])
Base.eltype(::TestRosenbrock1) = Float64

struct TestRosenbrock2 <: AbstractResidual end
NLLSModel.ndeps(::TestRosenbrock2) = static(1)
NLLSModel.nres(::TestRosenbrock2) = static(1)
NLLSModel.varindices(::TestRosenbrock2) = SVector(1)
NLLSModel.getvars(::TestRosenbrock2, vars::AbstractVector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::TestRosenbrock2, x) = SVector(10.0 * (x[2] - x[1]^2))
Base.eltype(::TestRosenbrock2) = Float64

# Multi-dimensional residual (nres = 2)
struct MultiDimRes <: AbstractResidual end
NLLSModel.ndeps(::MultiDimRes) = static(1)
NLLSModel.nres(::MultiDimRes) = static(2)
NLLSModel.varindices(::MultiDimRes) = SVector(1)
NLLSModel.getvars(::MultiDimRes, vars::AbstractVector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(::MultiDimRes, x) = SVector(x[1] - 2.0, 2.0 * x[2] - 4.0)
Base.eltype(::MultiDimRes) = Float64

# Scalar varindices (ndeps = 1, returning Int)
struct ScalarVarIndRes <: AbstractResidual end
NLLSModel.ndeps(::ScalarVarIndRes) = static(1)
NLLSModel.nres(::ScalarVarIndRes) = static(1)
NLLSModel.varindices(::ScalarVarIndRes) = 1
NLLSModel.getvars(::ScalarVarIndRes, vars::AbstractVector) = (vars[1]::Float64,)
NLLSModel.computeresidual(::ScalarVarIndRes, x) = SVector(x - 3.0)
Base.eltype(::ScalarVarIndRes) = Float64

# Pair difference residual
struct TestPairDiff <: AbstractResidual
    i::Int
    j::Int
    val::Float64
end
TestPairDiff(i, j) = TestPairDiff(i, j, 1.0)
NLLSModel.ndeps(::TestPairDiff) = static(2)
NLLSModel.nres(::TestPairDiff) = static(1)
NLLSModel.varindices(r::TestPairDiff) = SVector(r.i, r.j)
NLLSModel.getvars(r::TestPairDiff, vars::AbstractVector) = (vars[r.i]::SVector{1, Float64}, vars[r.j]::SVector{1, Float64})
NLLSModel.computeresidual(r::TestPairDiff, xi, xj) = SVector(xi[1] - xj[1] - r.val)
Base.eltype(::TestPairDiff) = Float64

# Dynamic vector residual
struct DynamicRes <: AbstractResidual
    dim::Int
end
NLLSModel.ndeps(::DynamicRes) = static(1)
NLLSModel.nres(d::DynamicRes) = d.dim
NLLSModel.varindices(::DynamicRes) = SVector(1)
NLLSModel.getvars(::DynamicRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(d::DynamicRes, x) = x .- 2.0
Base.eltype(::DynamicRes) = Float64

# Non-finite producing residual
struct NonFiniteRes <: AbstractResidual
    mode::Symbol
end
NLLSModel.ndeps(::NonFiniteRes) = static(1)
NLLSModel.nres(::NonFiniteRes) = static(1)
NLLSModel.varindices(::NonFiniteRes) = SVector(1)
NLLSModel.getvars(::NonFiniteRes, vars::AbstractVector) = (vars[1]::SVector{1, Float64},)
function NLLSModel.computeresidual(r::NonFiniteRes, x)
    if r.mode === :overflow
        return SVector(1e200 * x[1])
    elseif r.mode === :nan
        return SVector(NaN * x[1])
    else
        return SVector(Inf * x[1])
    end
end
Base.eltype(::NonFiniteRes) = Float64

# Generic wrapper to attach any robustifier to any residual
struct RobustWrapper{R<:AbstractResidual, K<:AbstractRobustifier} <: AbstractResidual
    res::R
    kernel::K
end
NLLSModel.ndeps(r::RobustWrapper) = ndeps(r.res)
NLLSModel.nres(r::RobustWrapper) = nres(r.res)
NLLSModel.varindices(r::RobustWrapper) = varindices(r.res)
NLLSModel.getvars(r::RobustWrapper, vars::AbstractVector) = getvars(r.res, vars)
NLLSModel.computeresidual(r::RobustWrapper, args...) = computeresidual(r.res, args...)
NLLSModel.robustkernel(r::RobustWrapper) = r.kernel
Base.eltype(r::RobustWrapper) = eltype(r.res)

# Kernel for testing clamp and domain error
struct MockClampingKernel <: AbstractRobustifier
    target_dc::Float64
end
NLLSModel.robustify(k::MockClampingKernel, s) = s
NLLSModel.robustifydcost(k::MockClampingKernel, s) = (s, k.target_dc, 0.0)

struct ZeroWeightKernel <: AbstractRobustifier end
NLLSModel.robustify(::ZeroWeightKernel, s) = 15.0
NLLSModel.robustifydcost(::ZeroWeightKernel, s) = (15.0, 0.0, 0.0)

# Adaptive residual definition for test
struct TestAdaptiveRes <: AbstractAdaptiveResidual
    varind::SVector{2, Int}
end
NLLSModel.ndeps(::TestAdaptiveRes) = static(2)
NLLSModel.nres(::TestAdaptiveRes) = static(1)
NLLSModel.varindices(r::TestAdaptiveRes) = r.varind
NLLSModel.getvars(r::TestAdaptiveRes, vars::Vector) = (vars[r.varind[1]]::ContaminatedGaussian{Float64}, vars[r.varind[2]]::Float64)
NLLSModel.computeresidual(::TestAdaptiveRes, x) = SVector(x - 1.0)
Base.eltype(::TestAdaptiveRes) = Float64

# Finite difference helper
function finite_diff_jac(res!, x, M; h=1e-7)
    N = length(x)
    J = Matrix{Float64}(undef, M, N)
    r_plus = Vector{Float64}(undef, M)
    r_minus = Vector{Float64}(undef, M)
    x_pert = copy(x)
    for j in 1:N
        x_pert[j] = x[j] + h
        res!(r_plus, x_pert)
        x_pert[j] = x[j] - h
        res!(r_minus, x_pert)
        x_pert[j] = x[j]
        J[:, j] .= (r_plus .- r_minus) ./ (2 * h)
    end
    return J
end

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
        r = zeros(flat.layout_tall.M)
        J = copy(flat.J_proto)
        c = flat.resjac!(r, J, flat.x0)
        J_fd = finite_diff_jac(flat.res!, flat.x0, flat.layout_tall.M)
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
        flat_sc = flat_optimvars_nls(prob_sc; robust=:irls)
        r_sc = zeros(1)
        J_sc = copy(flat_sc.J_proto)
        c_sc = flat_sc.resjac!(r_sc, J_sc, flat_sc.x0)
        J_sc_fd = finite_diff_jac(flat_sc.res!, flat_sc.x0, 1)
        @test isapprox(J_sc, J_sc_fd, atol=1e-5)
        @test isapprox(c_sc, 0.5 * dot(r_sc, r_sc); atol=1e-12)
    end

    @testset "6.1.2 Normal-Equation Equivalence" begin
        # Mode 1: robust = :none
        prob1 = Model(SVector{2, Float64}, Union{TestRosenbrock1, TestRosenbrock2})
        addvariable!(prob1, SVector(-1.2, 1.0))
        addcost!(prob1, TestRosenbrock1())
        addcost!(prob1, TestRosenbrock2())

        f_norm, fdf!_norm, x0_norm, _ = optimvars(prob1, nothing, static(1))
        g_norm = zeros(2)
        H_norm = zeros(2, 2)
        c_norm = fdf!_norm(g_norm, H_norm, x0_norm)

        f_tall, res!_tall, resjac!_tall, x0_tall, J_proto = optimvars_nls(prob1; robust=:none, numthreads=static(1))
        r_tall = zeros(2)
        J_tall = copy(J_proto)
        c_tall = resjac!_tall(r_tall, J_tall, x0_tall)

        @test isapprox(J_tall' * r_tall, g_norm; atol=1e-12)
        @test isapprox(J_tall' * J_tall, H_norm; atol=1e-12)
        @test isapprox(c_tall, c_norm; atol=1e-12)

        # Mode 2: robust = :irls
        prob2 = Model(Float64, RobustWrapper{ScalarVarIndRes, HuberKernel{Float64, Static.False}})
        addvariable!(prob2, 5.0)
        addcost!(prob2, RobustWrapper(ScalarVarIndRes(), HuberKernel(1.0)))
        _, fdf2, x02, _ = optimvars(prob2, nothing, static(1))
        g2 = zeros(1)
        H2 = zeros(1, 1)
        c2 = fdf2(g2, H2, x02)

        _, _, resjac2, _, J_proto2 = optimvars_nls(prob2; robust=:irls, numthreads=static(1))
        r2 = zeros(1)
        J2 = copy(J_proto2)
        c_tall2 = resjac2(r2, J2, x02)
        @test isapprox(J2' * r2, g2; atol=1e-12)
        @test isapprox(c_tall2, c2; atol=1e-12)

        # Mode 3: robust = :corrector across scales
        for scale in [1e-3, 1.0, 1e3]
            prob3 = Model(Float64, RobustWrapper{ScalarVarIndRes, HuberKernel{Float64, Static.True}})
            addvariable!(prob3, 5.0 * scale)
            addcost!(prob3, RobustWrapper(ScalarVarIndRes(), Huber2oKernel(2.0 * scale)))
            _, fdf3, x03, _ = optimvars(prob3, nothing, static(1))
            g3 = zeros(1)
            H3 = zeros(1, 1)
            c3 = fdf3(g3, H3, x03)

            _, _, resjac3, _, J_proto3 = optimvars_nls(prob3; robust=:corrector, numthreads=static(1))
            r3 = zeros(1)
            J3 = copy(J_proto3)
            c_tall3 = resjac3(r3, J3, x03)

            @test isapprox(J3' * r3, g3; atol=1e-9 * scale)
            @test isapprox(J3' * J3, H3; atol=1e-9 * scale)
            @test isapprox(c_tall3, c3; atol=1e-9 * scale^2)
        end
    end

    @testset "6.1.3 Reused Sparse J Zeroing & Cost Preservation" begin
        prob_zw = Model(Float64, RobustWrapper{ScalarVarIndRes, ZeroWeightKernel})
        addvariable!(prob_zw, 10.0)
        addcost!(prob_zw, RobustWrapper(ScalarVarIndRes(), ZeroWeightKernel()))

        flat = flat_optimvars_nls(prob_zw; robust=:irls, matrix_type=:sparse)
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

            flat = flat_optimvars_nls(prob; robust=:none)
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

        flat_clamp = flat_optimvars_nls(prob_clamp; robust=:irls)
        r = zeros(1)
        J = copy(flat_clamp.J_proto)

        @test clamped_count(flat_clamp) == 0
        c = flat_clamp.resjac!(r, J, flat_clamp.x0)
        @test clamped_count(flat_clamp) == 1
        @test r[1] == 0.0
        @test J[1, 1] == 0.0

        reset_clamped_count!(flat_clamp)
        @test clamped_count(flat_clamp) == 0

        # 2. Domain error: negative derivative beyond tolerance
        prob_domain = Model(Float64, RobustWrapper{ScalarVarIndRes, MockClampingKernel})
        addvariable!(prob_domain, 3.0)
        addcost!(prob_domain, RobustWrapper(ScalarVarIndRes(), MockClampingKernel(-0.1)))

        flat_domain = flat_optimvars_nls(prob_domain; robust=:irls)
        r_dom = zeros(1)
        J_dom = copy(flat_domain.J_proto)
        @test_throws DomainError flat_domain.resjac!(r_dom, J_dom, flat_domain.x0)
    end

    @testset "6.1.6 Multithreading Consistency" begin
        # 100 costs
        prob = Model(SVector{1, Float64}, TestPairDiff)
        for i in 1:101
            addvariable!(prob, SVector(Float64(i)))
        end
        for i in 1:100
            addcost!(prob, TestPairDiff(i, i + 1, 0.5))
        end

        f1, res1, resjac1, x01, J_proto1 = optimvars_nls(prob; robust=:none, numthreads=static(1))
        r1 = zeros(100)
        J1 = copy(J_proto1)
        c1 = resjac1(r1, J1, x01)

        f4, res4, resjac4, x04, J_proto4 = optimvars_nls(prob; robust=:none, numthreads=static(4), min_work_per_thread=10)
        r4 = zeros(100)
        J4 = copy(J_proto4)
        c4 = resjac4(r4, J4, x04)

        @test r1 == r4
        @test J1 == J4
        @test isapprox(c1, c4; atol=1e-12)
    end

    @testset "6.1.7 Fixed AbstractAdaptiveResidual Support" begin
        prob = Model(Union{ContaminatedGaussian{Float64}, Float64}, TestAdaptiveRes)
        kernel = ContaminatedGaussian(1.0, 5.0, 0.8)
        addvariable!(prob, kernel) # var 1 = adaptive kernel
        addvariable!(prob, 2.0)    # var 2 = value

        addcost!(prob, TestAdaptiveRes(SVector(1, 2)))

        # Unfixed adaptive variable must throw ArgumentError
        @test_throws ArgumentError optimvars_nls(prob; unfixed=nothing, robust=:irls)

        # Fixed adaptive variable (unfixed = 2) must succeed
        flat_adapt = flat_optimvars_nls(prob; unfixed=2, robust=:irls)
        r = zeros(1)
        J = copy(flat_adapt.J_proto)
        c = flat_adapt.resjac!(r, J, flat_adapt.x0)
        @test isfinite(c)
        @test isfinite(r[1])
        @test isfinite(J[1, 1])
    end

    @testset "6.1.8 Guards & Trait Checks" begin
        # 1. robust=:none with HuberKernel throws
        prob_huber = Model(Float64, RobustWrapper{ScalarVarIndRes, HuberKernel{Float64, Static.False}})
        addvariable!(prob_huber, 3.0)
        addcost!(prob_huber, RobustWrapper(ScalarVarIndRes(), HuberKernel(1.0)))
        @test_throws ArgumentError optimvars_nls(prob_huber; robust=:none)

        # 2. Scaled(NoRobust(), -1.0) throws
        prob_neg = Model(Float64, RobustWrapper{ScalarVarIndRes, Scaled{Float64, NoRobust}})
        addvariable!(prob_neg, 3.0)
        addcost!(prob_neg, RobustWrapper(ScalarVarIndRes(), Scaled(NoRobust(), -1.0)))
        @test_throws ArgumentError nls_callbacks(prob_neg)

        # 3. Non-additive variable type throws
        @test !isadditive(ZeroToInfScalar{Float64})
        @test_throws ArgumentError NLLSModel.check_additive_variable(ZeroToInfScalar(1.0))

        # 4. Dense J with NaNs has off-structure entries zeroed
        prob_dense = Model(SVector{2, Float64}, TestRosenbrock1)
        addvariable!(prob_dense, SVector(1.0, 2.0))
        addcost!(prob_dense, TestRosenbrock1())
        _, _, resjac_d, x0_d, _ = optimvars_nls(prob_dense; robust=:none, matrix_type=:dense)
        J_nan = fill(NaN, 1, 2)
        r_d = zeros(1)
        resjac_d(r_d, J_nan, x0_d)
        @test all(isfinite, J_nan)
    end
end

@testset "NLS Interface - NLPModels (NLLSNLSModel) & JSOSolvers" begin
    prob = Model(SVector{2, Float64}, Union{TestRosenbrock1, TestRosenbrock2})
    addvariable!(prob, SVector(-1.2, 1.0))
    addcost!(prob, TestRosenbrock1())
    addcost!(prob, TestRosenbrock2())

    nls = NLLSNLSModel(prob; name="RosenbrockNLS")
    @test nls isa AbstractNLSModel
    @test get_nvar(nls.meta) == 2
    @test get_nequ(nls.nls_meta) == 2
    @test get_x0(nls.meta) == [-1.2, 1.0]

    # NLPModels methods
    x0 = copy(nls.meta.x0)
    F = zeros(2)
    residual!(nls, x0, F)
    @test isapprox(F, [2.2, -4.4]; atol=1e-12)

    rows = zeros(Int, 4)
    cols = zeros(Int, 4)
    jac_structure_residual!(nls, rows, cols)
    @test rows == [1, 2, 1, 2]
    @test cols == [1, 1, 2, 2]

    vals = zeros(4)
    jac_coord_residual!(nls, x0, vals)
    @test isapprox(vals, [-1.0, 24.0, 0.0, 10.0]; atol=1e-12)

    Jv = zeros(2)
    jprod_residual!(nls, x0, [1.0, 0.5], Jv)
    @test isapprox(Jv, [-1.0, 29.0]; atol=1e-12)

    Jtv = zeros(2)
    jtprod_residual!(nls, x0, [1.0, 0.5], Jtv)
    @test isapprox(Jtv, [11.0, 5.0]; atol=1e-12)

    c = obj(nls, x0)
    @test isapprox(c, 12.1; atol=1e-12)

    g = zeros(2)
    grad!(nls, x0, g)
    @test isapprox(g, [-107.8, -44.0]; atol=1e-12)

    # Interleaved cache test
    reset!(nls)
    @test !nls.res_valid
    @test !nls.jac_valid
    residual!(nls, x0, F)
    @test nls.res_valid
    @test !nls.jac_valid
    jac_coord_residual!(nls, x0, vals)
    @test nls.res_valid
    @test nls.jac_valid
    c_cached = obj(nls, x0)
    @test isapprox(c_cached, 12.1; atol=1e-12)

    # Solve with JSOSolvers.trunk
    reset!(nls)
    stats_trunk = JSOSolvers.trunk(nls)
    @test stats_trunk.status == :first_order
    @test isapprox(stats_trunk.solution, [1.0, 1.0]; atol=1e-4)

    # Solve with JSOSolvers.tron
    reset!(nls)
    stats_tron = JSOSolvers.tron(nls)
    @test stats_tron.status == :first_order
    @test isapprox(stats_tron.solution, [1.0, 1.0]; atol=1e-4)

    # syncvars! round-trip
    syncvars!(nls, stats_trunk.solution)
    @test isapprox(prob.variables[1], SVector(1.0, 1.0); atol=1e-4)
end

@testset "NLS Interface - nls_callbacks & LeastSquaresOptim" begin
    # 1. Dense problem
    prob_dense = Model(SVector{2, Float64}, Union{TestRosenbrock1, TestRosenbrock2})
    addvariable!(prob_dense, SVector(-1.2, 1.0))
    addcost!(prob_dense, TestRosenbrock1())
    addcost!(prob_dense, TestRosenbrock2())

    cb = nls_callbacks(prob_dense)
    @test cb.nequ == 2
    @test cb.nvar == 2

    # Cache verification: jac! writes into J, subsequent res! at same x reuses cache
    J_test = copy(cb.J0)
    cb.jac!(J_test, cb.x0)
    r_test = zeros(2)
    cb.res!(r_test, cb.x0)
    @test isapprox(r_test, [2.2, -4.4]; atol=1e-12)

    lsp_dense = LeastSquaresOptim.LeastSquaresProblem(
        x = cb.x0,
        y = cb.r0,
        f! = cb.res!,
        g! = cb.jac!,
        J = cb.J0,
    )
    res_dense = LeastSquaresOptim.optimize!(lsp_dense, LeastSquaresOptim.LevenbergMarquardt())
    @test res_dense.converged
    @test isapprox(res_dense.minimizer, [1.0, 1.0]; atol=1e-4)

    # 2. Sparse problem with Scaled linear kernel
    prob_sp = Model(SVector{1, Float64}, RobustWrapper{TestPairDiff, Scaled{Float64, NoRobust}})
    for i in 1:6
        addvariable!(prob_sp, SVector(Float64(i)))
    end
    for i in 1:5
        addcost!(prob_sp, RobustWrapper(TestPairDiff(i, i + 1, 1.0), Scaled(NoRobust(), 2.0)))
    end

    cb_sp = nls_callbacks(prob_sp; matrix_type=:sparse)
    @test cb_sp.J0 isa SparseMatrixCSC

    lsp_sp = LeastSquaresOptim.LeastSquaresProblem(
        x = cb_sp.x0,
        y = cb_sp.r0,
        f! = cb_sp.res!,
        g! = cb_sp.jac!,
        J = cb_sp.J0,
    )
    res_sp = LeastSquaresOptim.optimize!(lsp_sp, LeastSquaresOptim.LevenbergMarquardt())
    @test res_sp.converged
end
