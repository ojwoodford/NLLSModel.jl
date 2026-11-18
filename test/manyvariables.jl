using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization

include("fixtures/loader.jl")
fixture("manyvariables")

# AN1: a cost with more than 5 variables, some of them fixed (no value dispatch on which: the variable flags are a run-time
# value), on both paths. Variables of static size (SVector{2}) and of run-time size (Vector{Float64} of length 2).

@testset "manyvariables.jl" begin
    nt = Threads.nthreads()
    opts = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200)
    @testset "a cost of $N variables, $V, unfixed = $unfixed" for N in (6, 10), V in (SVector{2, Float64}, Vector{Float64}),
            unfixed in (collect(2:N), collect(3:2:N), [N])
        model = manymodel(N, V)
        for numthreads in unique((1, nt))
            # The tall linearization equals the normal one, for the same unfixed variables
            _, _, resjac!, xt, J = optimvars_nls(model; unfixed, numthreads=static(numthreads), reduction=:none, storage=:dense, min_work=0, min_work_cost=0)
            r = zeros(size(J, 1))
            _, fdf!, xn, _ = optimvars(model; unfixed, numthreads=static(numthreads), reduction=:none, storage=:dense, min_work=0, min_work_cost=0)
            g, H = zeros(size(J, 2)), zeros(size(J, 2), size(J, 2))
            c = fdf!(g, H, xn)
            @test resjac!(r, J, xt) ≈ c rtol=1e-12
            @test J' * r ≈ g rtol=1e-9 atol=1e-12
            @test J' * J ≈ Symmetric(H) rtol=1e-9 atol=1e-12
            # The tall solve reaches the normal one's cost
            solved_hes = deepcopy(model)
            optimize!(solved_hes, opts; unfixed, numthreads=static(numthreads))
            solved_jac = deepcopy(model)
            optimize!(solved_jac, opts; formulation=:jacobian, unfixed, numthreads=static(numthreads))
            @test NLLSModel.cost(solved_hes) < NLLSModel.cost(model)
            @test NLLSModel.cost(solved_jac) ≈ NLLSModel.cost(solved_hes) rtol=1e-8
        end
    end
end
