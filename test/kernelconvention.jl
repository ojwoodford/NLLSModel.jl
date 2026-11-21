using NLLSModel, Test, Static, StaticArrays, Random

include("fixtures/loader.jl")
fixture("kernelconvention")

# The convention: ρ(s) = −2 log p(r), p a density over r ∈ ℝᵈ and s = ‖r‖²; an adaptive kernel returns it exactly
@testset "kernelconvention.jl" begin
    @testset "ContaminatedGaussian is normalized, d = $d, $(params)" for d in 1:3, params in ((0.5, 5.0, 0.4), (1.0, 10.0, 0.8))
        s1, s2, w = params
        @test_broken densityintegral(cgkernel(s1, s2, w, d), d, max(s1, s2)) ≈ 1.0 atol=1e-8
    end

    @testset "ContaminatedGaussian fit, d = 2" begin
        s = mixturesqnorms(MersenneTwister(2), 20_000, 2, 1.0, 10.0, 0.8)
        fitted = NLLSModel.params(fitkernel(cgkernel(0.5, 5.0, 0.5, 2), s))
        @test_broken fitted[1] ≈ 1.0 rtol=0.03
        @test_broken fitted[2] ≈ 10.0 rtol=0.03
    end
end
