using Test
using NLLSModel
using NLPModels
using Static, StaticArrays, SparseArrays, LinearAlgebra
import OptimizationVariableInterface: nvars, update, update!, syncvars!, ZeroToInfScalar
import JSOSolvers
import LeastSquaresOptim

include("fixtures/loader.jl")
fixture("nls_interface")

# The NLPModels and callback interfaces against third-party solvers, JSOSolvers and LeastSquaresOptim (the `integration`
# group).
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

    cb = NLLSModel.nls_callbacks(prob_dense)
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

    cb_sp = NLLSModel.nls_callbacks(prob_sp; storage=:sparse)
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
