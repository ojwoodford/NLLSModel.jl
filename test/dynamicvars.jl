using NLLSModels, Test, Static, Random, LinearAlgebra

struct LinearResidual <: NLLSModels.AbstractResidual
    y::Float64
    X::Vector{Float64}
end
NLLSModels.ndeps(::LinearResidual) = static(1) # Residual depends on 1 variables
NLLSModels.nres(::LinearResidual) = static(1) # Residual has length 1
NLLSModels.varindices(::LinearResidual) = 1
NLLSModels.getvars(::LinearResidual, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModels.computeresidual(res::LinearResidual, w) = res.X' * w - res.y
Base.eltype(::LinearResidual) = Float64

struct NormResidual <: NLLSModels.AbstractResidual
    len::Int
end
NLLSModels.ndeps(::NormResidual) = static(1) # Residual depends on 1 variables
NLLSModels.nres(res::NormResidual) = res.len
NLLSModels.varindices(::NormResidual) = 1
NLLSModels.getvars(::NormResidual, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModels.computeresidual(::NormResidual, w) = w
Base.eltype(::NormResidual) = Float64

@testset "dynamicvars.jl" begin
    # Generate some test data
    Random.seed!(1)
    X = normalize(randn(Int(ceil((1.0 + rand()) * 50))))
    n = length(X)

    # Create the problem
    problem = NLLSModels.NLLSProblem(Vector{Float64}, Union{LinearResidual, NormResidual})
    NLLSModels.addvariable!(problem, zeros(n))
    NLLSModels.addcost!(problem, LinearResidual(1.0, X))
    NLLSModels.addcost!(problem, NormResidual(n))

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
