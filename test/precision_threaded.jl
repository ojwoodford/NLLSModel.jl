using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization, BlockLinearSystems
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("optimization")

# The precision keyword on several threads (SP-N4): the per-thread copies of the system are in Float32 too and are
# summed in Float32, in another order than the Float64 fill's, so the Float32 fill equals the Float64 fill narrowed to
# 1e-5 relative (of the largest entry), not exactly.
@testset "precision_threaded.jl: a Float32 fill on $(Threads.nthreads()) threads" begin
    rng = Xoshiro(3)
    model = optlandmark_problem(6, 60)
    nt = static(Threads.nthreads())
    entries(H::SchurBlockHessian) = [H.R isa Matrix ? vec(H.R) : H.R.data; H.E_data; H.C_data]
    entries(H::SparseMatrixCSC) = nonzeros(H)
    entries(H::Matrix) = vec(H)
    for (red, st) in ((:none, :dense), (:none, :sparse), (:eliminate, :dense), (:eliminate, :sparse))
        f, fdf!, x, H64 = optimvars(model; reduction=red, storage=st, numthreads=nt, min_work=0)
        fdf32!, H32 = optimvars(model; reduction=red, storage=st, numthreads=nt, min_work=0, precision=Float32)[[2, 4]]
        n = OptimizationVariableInterface.nvars(x)
        H64 === nothing && (H64 = zeros(n, n))
        g64, g32 = zeros(n), zeros(n)
        @test fdf!(g64, H64, x) ≈ fdf32!(g32, H32, x) rtol=1e-12
        @test g32 ≈ g64 rtol=1e-12
        a64, a32 = entries(H64), entries(H32)
        @test maximum(abs.(Float64.(a32) .- a64)) <= 1e-5 * maximum(abs, a64)
        f, _, rj, x1, J64 = optimvars_nls(model; reduction=red, storage=st, numthreads=nt, min_work=0)
        _, _, rj32, _, J32 = optimvars_nls(model; reduction=red, storage=st, numthreads=nt, min_work=0, precision=Float32)
        r64, r32 = zeros(red === :none ? size(J64, 1) : 0), zeros(red === :none ? size(J64, 1) : 0)
        @test rj(r64, J64, x1) ≈ rj32(r32, J32, x1) rtol=1e-12
        J64 isa SchurBlockJacobian ? (@test J32.data == Float32.(J64.data) && J32.r == J64.r) : (@test J32 == Float32.(J64))   # each entry written once
    end
end
