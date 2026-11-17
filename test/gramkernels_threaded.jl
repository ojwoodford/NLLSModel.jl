using NLLSModel, Test, Static, StaticArrays, LinearAlgebra, Random
import SecondOrderOptimization: optimize!, OptimizationOptions

include("fixtures/loader.jl")
fixture("allocations", "gramkernels")

# The testsets of gramkernels.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "B1: a whole solve with (SVector, Matrix) costs, $nthreads thread(s)" for nthreads in (1, 4)
    options = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=50)
    hand, auto = procproblem(true, 300, 51), procproblem(false, 300, 51)
    optimize!(hand, options; numthreads=static(nthreads), min_work=0, min_work_cost=0)
    optimize!(auto, options; numthreads=static(nthreads), min_work=0, min_work_cost=0)
    @test NLLSModel.cost(hand) ≈ NLLSModel.cost(auto) rtol=1e-8
    @test hand.variables[1] ≈ auto.variables[1] atol=1e-6
end
