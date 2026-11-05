using Test
using NLLSModels
using NLPModels
using Static, StaticArrays, SparseArrays, LinearAlgebra
import OptimizationVariableInterface: nvars, syncvars!

# Define sample residuals for tests
struct Rosenbrock1 <: AbstractResidual end
NLLSModels.ndeps(::Rosenbrock1) = static(1)
NLLSModels.nres(::Rosenbrock1) = static(1)
NLLSModels.varindices(::Rosenbrock1) = SVector(1)
NLLSModels.getvars(::Rosenbrock1, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModels.computeresidual(::Rosenbrock1, x) = SVector(1.0 - x[1])
Base.eltype(::Rosenbrock1) = Float64

struct Rosenbrock2 <: AbstractResidual end
NLLSModels.ndeps(::Rosenbrock2) = static(1)
NLLSModels.nres(::Rosenbrock2) = static(1)
NLLSModels.varindices(::Rosenbrock2) = SVector(1)
NLLSModels.getvars(::Rosenbrock2, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModels.computeresidual(::Rosenbrock2, x) = SVector(10.0 * (x[2] - x[1]^2))
Base.eltype(::Rosenbrock2) = Float64

struct PairDiff <: AbstractResidual
    i::Int
    j::Int
    val::Float64
end
PairDiff(i, j) = PairDiff(i, j, 1.0)
NLLSModels.ndeps(::PairDiff) = static(2)
NLLSModels.nres(::PairDiff) = static(1)
NLLSModels.varindices(r::PairDiff) = SVector(r.i, r.j)
NLLSModels.getvars(r::PairDiff, vars::Vector) = (vars[r.i]::SVector{1, Float64}, vars[r.j]::SVector{1, Float64})
NLLSModels.computeresidual(r::PairDiff, xi, xj) = SVector(xi[1] - xj[1] - r.val)
Base.eltype(::PairDiff) = Float64

struct AnchorVal <: AbstractResidual
    val::Float64
end
AnchorVal() = AnchorVal(0.0)
NLLSModels.ndeps(::AnchorVal) = static(1)
NLLSModels.nres(::AnchorVal) = static(1)
NLLSModels.varindices(::AnchorVal) = SVector(1)
NLLSModels.getvars(::AnchorVal, vars::Vector) = (vars[1]::SVector{1, Float64},)
NLLSModels.computeresidual(r::AnchorVal, x1) = SVector(x1[1] - r.val)
Base.eltype(::AnchorVal) = Float64

@testset "NLPModelExt - Single-Variable Dense Model" begin
    prob = NLLSProblem(SVector{2, Float64}, Union{Rosenbrock1, Rosenbrock2})
    addvariable!(prob, SVector(-1.2, 1.0))
    addcost!(prob, Rosenbrock1())
    addcost!(prob, Rosenbrock2())

    # Construction
    nlp = NLLSModel(prob; name="Rosenbrock2D")
    @test nlp isa AbstractNLPModel
    @test get_nvar(nlp.meta) == 2
    @test get_nnzh(nlp.meta) == 3
    @test get_ncon(nlp.meta) == 0
    @test get_x0(nlp.meta) == [-1.2, 1.0]
    @test get_name(nlp.meta) == "Rosenbrock2D"

    # Objective
    c0 = obj(nlp, nlp.meta.x0)
    expected_c0 = cost(prob.variables, prob.costs)
    @test isapprox(c0, expected_c0, atol=1e-12)
    @test isapprox(c0, 12.1, atol=1e-12) # 0.5 * ( (1 - (-1.2))^2 + 100 * (1 - 1.44)^2 ) = 0.5 * (4.84 + 19.36) = 12.1
    @test neval_obj(nlp) == 1

    # Gradient
    g = grad(nlp, nlp.meta.x0)
    # J_1 = [-1, 0], r_1 = 2.2 => J_1^T r_1 = [-2.2, 0]
    # J_2 = [-20*x1, 10] = [24, 10], r_2 = -4.4 => J_2^T r_2 = [-105.6, -44.0]
    # sum = [-107.8, -44.0]
    @test isapprox(g, [-107.8, -44.0], atol=1e-12)
    @test neval_grad(nlp) == 1

    # Combined objgrad
    c_eval, g_eval = objgrad(nlp, nlp.meta.x0)
    @test isapprox(c_eval, c0, atol=1e-12)
    @test isapprox(g_eval, g, atol=1e-12)

    # Hessian structure
    rows, cols = hess_structure(nlp)
    @test rows == [1, 2, 2]
    @test cols == [1, 1, 2]

    # Hessian coordinates
    vals = hess_coord(nlp, nlp.meta.x0)
    # J^T J = [1 + 576, 240; 240, 100] = [577, 240; 240, 100]
    # Lower triangle: (1,1)=577, (2,1)=240, (2,2)=100
    @test isapprox(vals, [577.0, 240.0, 100.0], atol=1e-12)

    # Hessian sparse matrix
    H_sparse = hess(nlp, nlp.meta.x0)
    H_dense = Matrix(Symmetric(Matrix(H_sparse), :L))
    @test isapprox(H_dense, [577.0 240.0; 240.0 100.0], atol=1e-12)

    # Hessian-vector product
    v = [0.5, -0.2]
    Hv = hprod(nlp, nlp.meta.x0, v)
    @test isapprox(Hv, H_dense * v, atol=1e-12)

    # Weighted Hessian & Hprod
    vals_w = hess_coord(nlp, nlp.meta.x0; obj_weight=2.5)
    @test isapprox(vals_w, 2.5 .* vals, atol=1e-12)
    Hv_w = hprod(nlp, nlp.meta.x0, v; obj_weight=0.5)
    @test isapprox(Hv_w, 0.5 .* Hv, atol=1e-12)

    # 4-arg and 5-arg Lagrangian versions (with empty multipliers)
    y_empty = Float64[]
    vals_lag = zeros(3)
    NLPModels.hess_coord!(nlp, nlp.meta.x0, y_empty, vals_lag)
    @test isapprox(vals_lag, vals, atol=1e-12)

    Hv_lag = zeros(2)
    NLPModels.hprod!(nlp, nlp.meta.x0, y_empty, v, Hv_lag)
    @test isapprox(Hv_lag, Hv, atol=1e-12)

    # Reset
    reset!(nlp)
    @test neval_obj(nlp) == 0
    @test neval_grad(nlp) == 0
    @test neval_hess(nlp) == 0
    @test neval_hprod(nlp) == 0

    # Syncvars
    x_target = [1.0, 1.0]
    syncvars!(nlp, x_target)
    @test prob.variables[1] == SVector(1.0, 1.0)
    @test isapprox(cost(prob.variables, prob.costs), 0.0, atol=1e-20)
end

@testset "NLPModelExt - Multi-Variable Dense Model" begin
    prob = NLLSProblem(SVector{1, Float64}, Union{PairDiff, AnchorVal})
    for i in 1:5
        addvariable!(prob, SVector(0.0))
    end
    addcost!(prob, AnchorVal(0.0))
    for i in 1:4
        addcost!(prob, PairDiff(i, i+1, 2.0))
    end

    nlp = NLLSModel(prob)
    @test get_nvar(nlp.meta) == 5
    @test get_nnzh(nlp.meta) == 15 # 5*(6)/2 = 15 for dense
    @test get_x0(nlp.meta) == zeros(5)

    c0 = obj(nlp, nlp.meta.x0)
    @test isapprox(c0, cost(prob.variables, prob.costs), atol=1e-12)

    g = grad(nlp, nlp.meta.x0)
    @test length(g) == 5

    H = hess(nlp, nlp.meta.x0)
    v = ones(5)
    Hv = hprod(nlp, nlp.meta.x0, v)
    @test isapprox(Hv, Matrix(Symmetric(Matrix(H), :L)) * v, atol=1e-12)

    # Syncvars updates all variables in the problem
    x_sol = [0.0, -2.0, -4.0, -6.0, -8.0]
    syncvars!(nlp, x_sol)
    for i in 1:5
        @test prob.variables[i] == SVector(x_sol[i])
    end
    @test isapprox(cost(prob.variables, prob.costs), 0.0, atol=1e-20)
end

@testset "NLPModelExt - Multi-Variable Block-Sparse Model" begin
    N = 60 # >= 40 to trigger sparse linear system
    prob = NLLSProblem(SVector{1, Float64}, Union{PairDiff, AnchorVal})
    for i in 1:N
        addvariable!(prob, SVector(0.0))
    end
    addcost!(prob, AnchorVal(0.0))
    for i in 1:N-1
        addcost!(prob, PairDiff(i, i+1, 1.0))
    end

    nlp = NLLSModel(prob)
    @test get_nvar(nlp.meta) == N
    # Tridiagonal lower triangle: N diagonal entries + (N-1) subdiagonal entries = 2N - 1
    @test get_nnzh(nlp.meta) == 2 * N - 1
    @test nlp.h_nzindices !== nothing # Sparse branch

    # Check structure entries are in lower triangle
    rows, cols = hess_structure(nlp)
    @test length(rows) == 2 * N - 1
    @test all(rows .>= cols)

    # Verify Hessian values
    H_sparse = hess(nlp, nlp.meta.x0)
    @test nnz(H_sparse.data) == 2 * N - 1
    H_dense = Matrix(Symmetric(Matrix(H_sparse), :L))

    v = rand(N)
    Hv = hprod(nlp, nlp.meta.x0, v)
    @test isapprox(Hv, H_dense * v, atol=1e-12)

    # Syncvars
    x_opt = collect(0.0:-1.0:Float64(-(N-1)))
    syncvars!(nlp, x_opt)
    for i in 1:N
        @test isapprox(prob.variables[i][1], x_opt[i], atol=1e-12)
    end
    @test isapprox(cost(prob.variables, prob.costs), 0.0, atol=1e-20)
end

@testset "NLPModelExt - Unfixed Subset of Variables" begin
    prob = NLLSProblem(SVector{1, Float64}, PairDiff)
    addvariable!(prob, SVector(10.0))
    addvariable!(prob, SVector(0.0))
    addcost!(prob, PairDiff(1, 2, 3.0)) # x1 - x2 - 3 = 0 => x2 = 7

    # Variable 2 only
    nlp = NLLSModel(prob; unfixed=2)
    @test get_nvar(nlp.meta) == 1
    @test get_x0(nlp.meta) == [0.0]

    # Verify objective
    @test isapprox(obj(nlp, [0.0]), 0.5 * (10.0 - 0.0 - 3.0)^2, atol=1e-12)

    # Gradient w.r.t x2: r = 7, dr/dx2 = -1 => g = -7
    @test isapprox(grad(nlp, [0.0]), [-7.0], atol=1e-12)

    # Hessian w.r.t x2: (-1)^2 = 1
    @test isapprox(hess_coord(nlp, [0.0]), [1.0], atol=1e-12)

    # Syncvars updates only unfixed variable
    syncvars!(nlp, [7.0])
    @test prob.variables[1] == SVector(10.0)
    @test prob.variables[2] == SVector(7.0)
    @test isapprox(cost(prob.variables, prob.costs), 0.0, atol=1e-20)
end

@testset "NLPModelExt - AbstractNLPModel Conversion & Constructor" begin
    prob = NLLSProblem(SVector{2, Float64}, Union{Rosenbrock1, Rosenbrock2})
    addvariable!(prob, SVector(0.0, 0.0))
    addcost!(prob, Rosenbrock1())

    nlp1 = AbstractNLPModel(prob)
    @test nlp1 isa AbstractNLPModel

    nlp2 = convert(AbstractNLPModel, prob)
    @test nlp2 isa AbstractNLPModel
end
