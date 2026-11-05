using NLLSModel, Test, Static, StaticArrays, LinearAlgebra
using OptimizationVariableInterface: update!

const NDIMS = 3
struct LinearResidualStatic <: NLLSModel.AbstractResidual
    y::SVector{NDIMS, Float64}
    X::SMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS}
    varind::Int
end
NLLSModel.ndeps(::LinearResidualStatic) = static(1)
NLLSModel.nres(::LinearResidualStatic) = static(NDIMS)
NLLSModel.varindices(res::LinearResidualStatic) = res.varind
NLLSModel.getvars(res::LinearResidualStatic, vars::Vector) = (vars[res.varind]::SVector{NDIMS, Float64},)
NLLSModel.computeresidual(res::LinearResidualStatic, w) = res.X * w - res.y
Base.eltype(::LinearResidualStatic) = Float64

struct LinearResidualDynamic <: NLLSModel.AbstractResidual
    y::Vector{Float64}
    X::Matrix{Float64}
    varind::Int
end
NLLSModel.ndeps(::LinearResidualDynamic) = static(1)
NLLSModel.nres(res::LinearResidualDynamic) = length(res.y)
NLLSModel.varindices(res::LinearResidualDynamic) = res.varind
NLLSModel.getvars(res::LinearResidualDynamic, vars::Vector) = (vars[res.varind]::Vector{Float64},)
NLLSModel.computeresidual(res::LinearResidualDynamic, w) = res.X * w - res.y
Base.eltype(::LinearResidualDynamic) = Float64

struct LinearCostStatic <: NLLSModel.AbstractCost
    y::SVector{NDIMS, Float64}
    varind::Int
end
NLLSModel.ndeps(::LinearCostStatic) = static(1)
NLLSModel.varindices(cost::LinearCostStatic) = cost.varind
NLLSModel.getvars(cost::LinearCostStatic, vars::Vector) = (vars[cost.varind]::SVector{NDIMS, Float64},)
NLLSModel.computecost(cost::LinearCostStatic, w) = cost.y' * w
Base.eltype(::LinearCostStatic) = Float64

struct LinearCostDynamic <: NLLSModel.AbstractCost
    y::Vector{Float64}
    varind::Int
end
NLLSModel.ndeps(::LinearCostDynamic) = static(1)
NLLSModel.varindices(cost::LinearCostDynamic) = cost.varind
NLLSModel.getvars(cost::LinearCostDynamic, vars::Vector) = (vars[cost.varind]::Vector{Float64},)
NLLSModel.computecost(cost::LinearCostDynamic, w) = cost.y' * w
Base.eltype(::LinearCostDynamic) = Float64

@testset "nonsquaredcost.jl" begin
    # Generate some test data
    X = randn(SMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS})
    y = randn(SVector{NDIMS, Float64})
    solution = (X' * X) \ ((X' - I) * y)

    # Create the problem
    problem = NLLSModel.NLLSModel(Union{SVector{NDIMS, Float64}, Vector{Float64}}, Union{LinearResidualStatic, LinearCostStatic, LinearResidualDynamic, LinearCostDynamic})
    NLLSModel.addvariable!(problem, zeros(SVector{NDIMS, Float64}))
    NLLSModel.addcost!(problem, LinearResidualStatic(y, X, 1))
    NLLSModel.addcost!(problem, LinearCostStatic(y, 1))
    NLLSModel.addvariable!(problem, zeros(NDIMS))
    NLLSModel.addcost!(problem, LinearResidualDynamic(Vector(y), Matrix(X), 2))
    NLLSModel.addcost!(problem, LinearCostDynamic(Vector(y), 2))

    # Evaluate via optimvars (all variables)
    f, fdf!, x0, H = optimvars(problem)
    @test H === nothing
    g = zeros(2 * NDIMS)
    H_mat = zeros(2 * NDIMS, 2 * NDIMS)
    c = fdf!(g, H_mat, x0)
    @test c ≈ f(x0)

    step = -(H_mat \ g)
    x_new = deepcopy(x0)
    update!(x_new, x0, step)
    @test x_new[1] ≈ solution
    @test x_new[2] ≈ solution

    # Single-variable optimvars (static variable 1)
    sub1 = subproblem(problem, 1)
    f1, fdf1!, x1, H1 = optimvars(sub1, 1)
    @test H1 === nothing
    g1 = zeros(MVector{NDIMS, Float64})
    H1_mat = zeros(MMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS})
    c1 = fdf1!(g1, H1_mat, x1)
    @test c1 ≈ f1(x1)
    @test (x1 - H1_mat \ g1) ≈ solution
end
