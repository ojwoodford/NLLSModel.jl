using NLLSModel, Test, Static, Random, LinearAlgebra

struct LinearResidual <: NLLSModel.AbstractResidual
    y::Float64
    X::Vector{Float64}
end
NLLSModel.ndeps(::LinearResidual) = static(1) # Residual depends on 1 variables
NLLSModel.nres(::LinearResidual) = static(1) # Residual has length 1
NLLSModel.varindices(::LinearResidual) = 1
NLLSModel.getvars(::LinearResidual, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(res::LinearResidual, w) = res.X' * w - res.y
Base.eltype(::LinearResidual) = Float64

struct NormResidual <: NLLSModel.AbstractResidual
    len::Int
end
NLLSModel.ndeps(::NormResidual) = static(1) # Residual depends on 1 variables
NLLSModel.nres(res::NormResidual) = res.len
NLLSModel.varindices(::NormResidual) = 1
NLLSModel.getvars(::NormResidual, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(::NormResidual, w) = w
Base.eltype(::NormResidual) = Float64

@testset "dynamicvars.jl" begin
    # Generate some test data
    Random.seed!(1)
    X = normalize(randn(Int(ceil((1.0 + rand()) * 50))))
    n = length(X)

    # Create the problem
    problem = Model(Vector{Float64}, Union{LinearResidual, NormResidual})
    NLLSModel.addvariable!(problem, zeros(n))
    NLLSModel.addcost!(problem, LinearResidual(1.0, X))
    NLLSModel.addcost!(problem, NormResidual(n))

    # Build optimization variables via optimvars
    f, fdf!, x0, H = optimvars(problem)
    @test H === nothing
    @test x0 == zeros(n)
    @test f(x0) ≈ cost(problem)

    g = zeros(n)
    H_mat = zeros(n, n)
    c = fdf!(g, H_mat, x0)
    @test c ≈ f(x0)

    # Exact Newton step on quadratic problem
    Y = x0 - H_mat \ g
    @test X' * Y ≈ norm(Y)
end
