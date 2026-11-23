using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("frontdoor", "optimization")

# The testsets of frontdoor.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "frontdoor.jl, threaded" begin
    # For one variable, the H that optimvars returns has the types of the linear system's own storage, so buffers built
    # from it (H by deepcopy, g by `zero(H[:, 1])`) take fdf!'s direct-write branch: the costs accumulate straight into
    # them, with no internal accumulation and no copy. Other buffer types take the fallback, with the same result.
    @testset "single-variable prototype: $name" for (name, V, n, nt) in (
            ("static 3", SVector{3, Float64}, 3, 1), ("static 16", SVector{16, Float64}, 16, 1),
            ("static 3, shared", SVector{3, Float64}, 3, 4), ("static 17", SVector{17, Float64}, 17, 1),
            ("dynamic 3", Vector{Float64}, 3, 1), ("dynamic 3, shared", Vector{Float64}, 3, 4))
        model = linsingleproblem(V, n, 11)
        _, fdf!, x0, H = optimvars(model; numthreads=static(nt), min_work=0) # min_work=0: share the system when nt > 1
        @test (fdf!.linsystem isa NLLSModel.LinearSystemShared) == (nt > 1)
        isstatic = V <: StaticVector && n <= 16
        @test H isa (isstatic ? MMatrix{n, n, Float64, n * n} : Matrix{Float64})
        @test size(H) == (n, n)
        @test all(iszero, H)
        Hd = deepcopy(H)
        gd = zero(H[:, 1])
        @test gd isa (isstatic ? MVector{n, Float64} : Vector{Float64})
        internal = NLLSModel.getA(fdf!.linsystem)
        fill!(internal, 7.0)
        c = fdf!(gd, Hd, x0)
        @test all(==(7.0), internal) # The direct branch never touches the internal storage
        @test singlefdf_allocs(fdf!, gd, Hd, x0) == 0 # Threaded too, when the suite runs on several threads
        # The fallback: buffers of another type than the system's
        gf, Hf = isstatic ? (zeros(n), zeros(n, n)) : (view(zeros(n), :), view(zeros(n, n), :, :))
        cf = fdf!(gf, Hf, x0)
        # The fallback: a static system on one thread writes from its values (internal storage untouched, item 19);
        # otherwise it accumulates internally, then copies
        @test all(==(7.0), internal) == (isstatic && nt == 1)
        @test cf == c
        @test gf == gd
        @test Hf == Hd
    end
end
