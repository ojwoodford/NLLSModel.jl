using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization

# Z2: formulation = :tall with every iterator the optimizer's Jacobian path accepts. Needs tallproblems.jl and sameoptimum and
# TallOptions from tallformulation.jl.

@testset "talliterators.jl" begin
    @testset "same optimum as :normal with $iterator, $name" for iterator in (:levenbergmarquardt_dtd, :dogleg, :newton), (name, make) in (("curve fit", () -> curvefit(outliers=2)), ("Huber", () -> curvefit(huber=true, outliers=4)), ("Huber2o", () -> curvefit(huber=:second, outliers=4)), ("mixed variables", () -> posemodel()))
        ok, ct, cn = sameoptimum(make(), OptimizationOptions(iterator=iterator, maxiters=300))
        @test ok
    end
end
