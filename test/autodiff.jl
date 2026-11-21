using NLLSModel, Test, Static, StaticArrays, Random

include("fixtures/loader.jl")
fixture("allocations", "autodiff")

# Autodiff of a cost whose sizes are all static but whose Jacobian is above MAX_STATIC_JAC (so not built as a static
# matrix): the Jacobian is exact and nothing allocates per call, with fewer and with more inputs than ForwardDiff's
# default chunk (12).

@testset "autodiff.jl" begin
    rng = MersenneTwister(4)
    # (rows, size of x, size of y): 12 inputs, 13 inputs, and the most a static evaluation takes (MAX_STATIC_VAR)
    @testset "static sizes, Jacobian above MAX_STATIC_JAC: $M × ($N1 + $N2)" for (M, N1, N2) in ((12, 3, 9), (10, 3, 10), (8, 32, 32))
        @test M * (N1 + N2) > NLLSModel.MAX_STATIC_JAC && N1 + N2 <= NLLSModel.MAX_STATIC_VAR
        cost = WideLinear{N1, N2}(randn(rng, SMatrix{M, N1 + N2, Float64}), randn(rng, SVector{M, Float64}))
        x, y = randn(rng, SVector{N1, Float64}), randn(rng, SVector{N2, Float64})
        res, jac = computeresjac(static(3), cost, x, y)
        @test jac == cost.A
        @test res ≈ cost.A * vcat(x, y) - cost.b
        # Only y unfixed
        res2, jac2 = computeresjac(static(2), cost, x, y)
        @test jac2 == cost.A[:, N1+1:end] && res2 ≈ res
        # Nothing allocates per call: the Jacobian, and the cost, gradient and Hessian built from it
        @test allocs_warm(computeresjac, static(3), cost, x, y) == 0
        @test allocs_warm(NLLSModel.computerescostgradhess, static(6), cost, NoRobust(), (x, y)) == 0
    end
    # A residual or cost without Base.eltype: an ArgumentError on each autodiff path (AN8)
    eltypemsg(f) = try f(); "no error" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError: $(typeof(e))" end
    for (path, r, x) in (("static", NoEltype(static(2)), SVector(1.0, 2.0)),
                         ("wide", NoEltype(static(60)), randn(rng, SVector{60, Float64})),
                         ("dynamic", NoEltype(2), [1.0, 2.0]))
        @testset "computeresjac, $path" begin
            @test occursin("Define Base.eltype()", eltypemsg(() -> computeresjac(static(1), r, x)))
        end
    end
    @test occursin("Define Base.eltype()", eltypemsg(() -> NLLSModel.computecostgradhess(static(1), NoEltypeCost(), [1.0, 2.0])))
end
