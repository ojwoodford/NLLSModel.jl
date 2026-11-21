using Test
using NLLSModel
using NLPModels
using Static, StaticArrays, SparseArrays, LinearAlgebra
import OptimizationVariableInterface: nvars, update, update!, syncvars!, ZeroToInfScalar

include("fixtures/loader.jl")
fixture("nls_interface")

# The testsets of nls_interface.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "NLS Interface - Core Functors & Traits, threaded" begin
    @testset "6.1.6 Multithreading Consistency" begin
        # 100 costs
        prob = Model(SVector{1, Float64}, TestPairDiff)
        for i in 1:101
            addvariable!(prob, SVector(Float64(i)))
        end
        for i in 1:100
            addcost!(prob, TestPairDiff(i, i + 1, 0.5))
        end

        f1, res1, resjac1, x01, J_proto1 = optimvars_nls(prob, numthreads=static(1))
        r1 = zeros(100)
        J1 = copy(J_proto1)
        c1 = resjac1(r1, J1, x01)

        f4, res4, resjac4, x04, J_proto4 = optimvars_nls(prob, numthreads=static(4), min_work=10)
        r4 = zeros(100)
        J4 = copy(J_proto4)
        c4 = resjac4(r4, J4, x04)

        @test r1 == r4
        @test J1 == J4
        @test isapprox(c1, c4; atol=1e-12)
    end
end
