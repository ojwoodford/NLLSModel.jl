using NLLSModels, Test, Static, StaticArrays, LinearAlgebra

const NDIMS = 3
struct LinearResidualStatic <: NLLSModels.AbstractResidual
    y::SVector{NDIMS, Float64}
    X::SMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS}
    varind::Int
end
NLLSModels.ndeps(::LinearResidualStatic) = static(1)
NLLSModels.nres(::LinearResidualStatic) = static(NDIMS)
NLLSModels.varindices(res::LinearResidualStatic) = res.varind
NLLSModels.getvars(res::LinearResidualStatic, vars::Vector) = (vars[res.varind]::SVector{NDIMS, Float64},)
NLLSModels.computeresidual(res::LinearResidualStatic, w) = res.X * w - res.y
Base.eltype(::LinearResidualStatic) = Float64

struct LinearResidualDynamic <: NLLSModels.AbstractResidual
    y::Vector{Float64}
    X::Matrix{Float64}
    varind::Int
end
NLLSModels.ndeps(::LinearResidualDynamic) = static(1)
NLLSModels.nres(res::LinearResidualDynamic) = length(res.y)
NLLSModels.varindices(res::LinearResidualDynamic) = res.varind
NLLSModels.getvars(res::LinearResidualDynamic, vars::Vector) = (vars[res.varind]::Vector{Float64},)
NLLSModels.computeresidual(res::LinearResidualDynamic, w) = res.X * w - res.y
Base.eltype(::LinearResidualDynamic) = Float64

struct LinearCostStatic <: NLLSModels.AbstractCost
    y::SVector{NDIMS, Float64}
    varind::Int
end
NLLSModels.ndeps(::LinearCostStatic) = static(1)
NLLSModels.varindices(cost::LinearCostStatic) = cost.varind
NLLSModels.getvars(cost::LinearCostStatic, vars::Vector) = (vars[cost.varind]::SVector{NDIMS, Float64},)
NLLSModels.computecost(cost::LinearCostStatic, w) = cost.y' * w
Base.eltype(::LinearCostStatic) = Float64

struct LinearCostDynamic <: NLLSModels.AbstractCost
    y::Vector{Float64}
    varind::Int
end
NLLSModels.ndeps(::LinearCostDynamic) = static(1)
NLLSModels.varindices(cost::LinearCostDynamic) = cost.varind
NLLSModels.getvars(cost::LinearCostDynamic, vars::Vector) = (vars[cost.varind]::Vector{Float64},)
NLLSModels.computecost(cost::LinearCostDynamic, w) = cost.y' * w
Base.eltype(::LinearCostDynamic) = Float64

@testset "nonsquaredcost.jl" begin
    # Generate some test data
    X = randn(SMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS})
    y = randn(SVector{NDIMS, Float64})
    solution = (X' * X) \ ((X' - I) * y)

    # Create the problem
    problem = NLLSModels.NLLSProblem(Union{SVector{NDIMS, Float64}, Vector{Float64}}, Union{LinearResidualStatic, LinearCostStatic, LinearResidualDynamic, LinearCostDynamic})
    NLLSModels.addvariable!(problem, zeros(SVector{NDIMS, Float64}))
    NLLSModels.addcost!(problem, LinearResidualStatic(y, X, 1))
    NLLSModels.addcost!(problem, LinearCostStatic(y, 1))
    NLLSModels.addvariable!(problem, zeros(NDIMS))
    NLLSModels.addcost!(problem, LinearResidualDynamic(Vector(y), Matrix(X), 2))
    NLLSModels.addcost!(problem, LinearCostDynamic(Vector(y), 2))

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
