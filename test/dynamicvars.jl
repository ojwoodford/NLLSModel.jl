using NLLSModel, Test, Static, Random, LinearAlgebra

include("fixtures/dynamicvars.jl")

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
    @test H isa Matrix{Float64} && size(H) == (n, n) # One dynamic variable: a dense prototype
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
