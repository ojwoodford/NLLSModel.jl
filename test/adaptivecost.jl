using NLLSModel, Test, Static, StaticArrays, Random
using OptimizationVariableInterface: update!, update

include("fixtures/loader.jl")
fixture("adaptivecost", "robustfit")

@testset "adaptivecost.jl" begin
    # Create the problem
    problem = Model(Union{NLLSModel.ContaminatedGaussian{Float64, 1}, Float64}, SimpleResidual)
    NLLSModel.addvariable!(problem, ContaminatedGaussian(0.5, 5.0, 0.4, static(1)))
    NLLSModel.addvariable!(problem, 0.)
    NLLSModel.addvariable!(problem, 0.)
    Random.seed!(1)
    points = vcat(randn(800), randn(200) * 10.0)
    for p in points
        NLLSModel.addcost!(problem, SimpleResidual(p - 1, 2))
        NLLSModel.addcost!(problem, SimpleResidual(p + 1, 3))
    end

    startcost = NLLSModel.cost(problem)
    optimizerobustifier!(problem, 1)
    @test NLLSModel.cost(problem) < startcost

    # Test optimvars with unfixed mask (optimizing only variables 2 and 3)
    unfixed = SVector(1, 2, 3) .> 1
    f_sub, fdf_sub!, x0_sub, H_sub = optimvars(problem; unfixed=unfixed)
    @test length(x0_sub) == 2
    @test H_sub === nothing
    g = zeros(2)
    H_mat = zeros(2, 2)
    for iter in 1:15
        fdf_sub!(g, H_mat, x0_sub)
        step = -(H_mat \ g)
        update!(x0_sub, x0_sub, step)
        f_sub(x0_sub)
        optimizerobustifier!(problem, 1)
    end
    @test isapprox(NLLSModel.params(problem.variables[1]), SVector(1.0, 10.0, 0.8); rtol=0.1)
    @test isapprox(problem.variables[2], -1.0; rtol=0.1)
    @test isapprox(problem.variables[3], 1.0; rtol=0.1)
end

@testset "ContaminatedGaussian construction" begin
    @test NLLSModel.params(ContaminatedGaussian(2.0, 5.0, 0.9, static(1))) ≈ SVector(2.0, 5.0, 0.9)
    # The parameters are kept in the order given, and the density is symmetric under exchanging the components
    @test NLLSModel.params(ContaminatedGaussian(5.0, 2.0, 0.1, static(1))) ≈ SVector(5.0, 2.0, 0.1)
    for d in 1:3, s in (0.0, 0.3, 4.0, 60.0)
        @test robustify(ContaminatedGaussian(2.0, 5.0, 0.9, static(d)), s) ≈ robustify(ContaminatedGaussian(5.0, 2.0, 0.1, static(d)), s)
    end
    # d is a StaticInt of at least 1
    @test_throws ArgumentError ContaminatedGaussian(2.0, 5.0, 0.9, 1)
    @test_throws ArgumentError ContaminatedGaussian(2.0, 5.0, 0.9, static(0))
    @test_throws ArgumentError ContaminatedGaussian(NLLSModel.ZeroToInfScalar(0.5), NLLSModel.ZeroToInfScalar(0.2), NLLSModel.ZeroToOneScalar(0.9), 1)
    err = try ContaminatedGaussian(2.0, 5.0, 0.9, 2); nothing catch e e end
    @test occursin("static(2)", err.msg)
    # Arguments are promoted to one float type
    @test ContaminatedGaussian(2, 5.0f0, 0.5f0, static(1)) isa ContaminatedGaussian{Float32, 1}
end

@testset "alternation from a callback, Hessian path, $(nameof(typeof(kernel))), $iterator, $nvar unfixed" for
        kernel in (ContaminatedGaussian(0.5, 5.0, 0.5, static(2)), GaussianPlusUniform(0.5, 0.5, static(2); volume=1600.0)),
        iterator in (:levenbergmarquardt, :dogleg), nvar in (2, 1)
    model, truths = alternationmodel(kernel; nvar)
    result, given, before, returned = alternate!(model, OptimizationOptions(; iterator, callbackchangesstate=true); formulation=:hessian)
    @test isapprox(NLLSModel.params(model.variables[1]), alternationtruth(kernel); rtol=0.1)
    for (i, m) in enumerate(truths)
        @test isapprox(model.variables[i + 1], m; rtol=0.1)
    end
    @test !isempty(returned) && all(diff(returned) .<= 0) # The costs the callback returns never increase
    @test given ≈ before rtol=1e-12 # At the callback the model holds the point the optimizer evaluated (to rounding: two sums)
end

@testset "a callback that changes the cost needs callbackchangesstate" begin
    model, _ = alternationmodel(GaussianPlusUniform(0.5, 0.5, static(2); volume=1600.0))
    err = try alternate!(model, OptimizationOptions(); formulation=:hessian); nothing catch e e end
    @test err isa ArgumentError && occursin("callbackchangesstate", err.msg)
end

@testset "joint optimization of a GaussianPlusUniform and the variables, Hessian path" begin
    model, truths = alternationmodel(GaussianPlusUniform(1.1, 0.75, static(2); volume=1600.0); start=SVector(0.1, -0.1))
    result = optimize!(model, OptimizationOptions(maxiters=200); numthreads=static(1))
    @test isapprox(NLLSModel.params(model.variables[1]), SVector(1.0, 0.8); rtol=0.1)
    for (i, m) in enumerate(truths)
        @test isapprox(model.variables[i + 1], m; rtol=0.1)
    end
end
