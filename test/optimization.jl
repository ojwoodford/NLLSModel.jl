using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("allocations", "optimization", "optimizeba")

@testset "optimization.jl" begin
    # Every linear system and iterator takes steps that reduce the cost of a noise-free bundle adjustment problem
    Random.seed!(1)
    @testset "dense and sparse, $iterator" for iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd, :newton)
        for (ncams, nlms) in ((3, 5), (10, 60))
            problem = perturb_ba_problem(create_ba_problem(ncams, nlms, 0.5), 0.01, 0.0)
            c0, c1, _ = solve(problem; iterator=iterator)
            @test c1 < 1e-3 * c0
        end
    end

    @testset "Schur, $storage, $iterator" for storage in (:dense, :sparse), iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd)
        problem = optlandmark_problem(4, 20)
        c0, c1, _ = solve(problem; iterator=iterator, reduction=:eliminate, storage=storage)
        @test c1 < 1e-3 * c0
    end

    # Dense and sparse Schur reach the same solution
    Random.seed!(4)
    p1 = optlandmark_problem(4, 20)
    p2 = deepcopy(p1)
    _, c_dense, _ = solve(p1; reduction=:eliminate, storage=:dense)
    _, c_sparse, _ = solve(p2; reduction=:eliminate, storage=:sparse)
    @test c_dense ≈ c_sparse atol=1e-8
end

@testset "Schur pattern is handed to the linear system" begin
    Random.seed!(6)
    prob = optlandmark_problem(4, 20)
    _, _, _, H = optimvars(prob; numthreads=static(1), reduction=:eliminate, storage=:sparse)
    @test H.pattern !== nothing
    @test size(H.pattern.S_sparse, 1) == sum(BlockLinearSystems.blocksizes(H.structure)[1:BlockLinearSystems.nretained(H.structure)])
    _, _, _, Hd = optimvars(prob; numthreads=static(1), reduction=:eliminate, storage=:dense)
    @test Hd.pattern === nothing
    # The solver accepts it and optimizes
    c0, c1, _ = solve(optlandmark_problem(4, 20); reduction=:eliminate, storage=:sparse)
    @test c1 < 1e-3 * c0
end
