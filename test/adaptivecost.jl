using NLLSModels, Test, Static, StaticArrays, Random
using OptimizationVariableInterface: update!

struct SimpleResidual <: NLLSModels.AbstractAdaptiveResidual
    data::Float64
    varind::Int
end
NLLSModels.ndeps(::SimpleResidual) = static(2)
NLLSModels.nres(::SimpleResidual) = static(1)
NLLSModels.varindices(res::SimpleResidual) = SVector(1, res.varind)
NLLSModels.getvars(res::SimpleResidual, vars::Vector) = (vars[1]::NLLSModels.ContaminatedGaussian{Float64}, vars[res.varind]::Float64)
NLLSModels.computeresidual(res::SimpleResidual, mean) = mean - res.data
NLLSModels.computeresjac(varflags, res::SimpleResidual, mean) = mean - res.data, SVector(one(mean))'
Base.eltype(::SimpleResidual) = Float64

function optimisekernel!(problem, vars=problem.variables)
    # Compute the squared errors
    squarederrors = [NLLSModels.computeresidual(res, vars[res.varind]) ^ 2 for res in get(problem.costs, SimpleResidual)]
    # Optimize the kernel parameters
    vars[1] = NLLSModels.optimize(vars[1], squarederrors)
    return nothing
end

@testset "adaptivecost.jl" begin
    # Create the problem
    problem = NLLSModels.NLLSProblem(Union{NLLSModels.ContaminatedGaussian{Float64}, Float64}, SimpleResidual)
    NLLSModels.addvariable!(problem, ContaminatedGaussian(0.5, 5.0, 0.4))
    NLLSModels.addvariable!(problem, 0.)
    NLLSModels.addvariable!(problem, 0.)
    Random.seed!(1)
    points = vcat(randn(800), randn(200) * 10.0)
    for p in points
        NLLSModels.addcost!(problem, SimpleResidual(p - 1, 2))
        NLLSModels.addcost!(problem, SimpleResidual(p + 1, 3))
    end

    startcost = NLLSModels.cost(problem)
    optimisekernel!(problem)
    @test NLLSModels.cost(problem) < startcost

    # Test optimvars with unfixed mask (optimizing only variables 2 and 3)
    unfixed = SVector(1, 2, 3) .> 1
    f_sub, fdf_sub!, x0_sub, H_sub = optimvars(problem, unfixed)
    @test length(x0_sub) == 2
    @test H_sub === nothing
    g = zeros(2)
    H_mat = zeros(2, 2)
    for iter in 1:15
        fdf_sub!(g, H_mat, x0_sub)
        step = -(H_mat \ g)
        update!(x0_sub, x0_sub, step)
        f_sub(x0_sub)
        optimisekernel!(problem)
    end
    @test isapprox(NLLSModels.params(problem.variables[1]), SVector(1.0, 10.0, 0.8); rtol=0.1)
    @test isapprox(problem.variables[2], -1.0; rtol=0.1)
    @test isapprox(problem.variables[3], 1.0; rtol=0.1)
end
