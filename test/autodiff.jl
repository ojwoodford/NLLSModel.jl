using NLLSModel, Test, Static, StaticArrays, Random

# Autodiff of a cost whose sizes are all static but whose Jacobian is above MAX_STATIC_JAC (so not built as a static
# matrix): the Jacobian is exact and nothing allocates per call, with fewer and with more inputs than ForwardDiff's
# default chunk (12). Needs allocs_warm from runtests.jl.

# r = A [x; y] - b, for an x of N1 and a y of N2 dimensions, M rows
struct WideLinear{M, N1, N2, L, ML} <: NLLSModel.AbstractResidual
    A::SMatrix{M, L, Float64, ML}
    b::SVector{M, Float64}
end
WideLinear{N1, N2}(A::SMatrix{M, L, Float64, ML}, b) where {M, N1, N2, L, ML} = (@assert L == N1 + N2; WideLinear{M, N1, N2, L, ML}(A, b))
NLLSModel.ndeps(::WideLinear) = static(2)
NLLSModel.nres(::WideLinear{M}) where M = static(M)
NLLSModel.varindices(::WideLinear) = SVector(1, 2)
NLLSModel.getvars(::WideLinear{M, N1, N2}, vars::AbstractVector) where {M, N1, N2} = (vars[1]::SVector{N1, Float64}, vars[2]::SVector{N2, Float64})
NLLSModel.computeresidual(r::WideLinear, x, y) = r.A * vcat(x, y) - r.b
Base.eltype(::WideLinear) = Float64

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
end
