using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization

include("fixtures/loader.jl")
fixture("modelworkspace")

# The testsets of modelworkspace.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4
# threads): a reused setup on several threads, where the setup holds a shared system (and, with elimination, a thread
# plan). `min_work = 0` and `min_work_cost = 0` use every thread there is work for.
@testset "modelworkspace.jl, threaded" begin
    @testset "the same solution as without a workspace, on 4 threads: $name" for (name, make, kw) in (
            ("single", () -> linsingleproblem(SVector{3, Float64}, 3, 11), (;)),
            ("dense", () -> staticchain(8), (;)),
            ("sparse", () -> staticchain(30), (;)),
            ("Schur", () -> (Random.seed!(3); optlandmark_problem(4, 20)), (;)),
            ("Jacobian", () -> staticchain(8), (formulation=:jacobian,)),
            ("Jacobian, eliminated", () -> (Random.seed!(3); optlandmark_problem(4, 20)), (formulation=:jacobian,)))
        kw = merge(kw, (numthreads=static(4), min_work=0, min_work_cost=0))
        model = make()
        start = deepcopy(model.variables)
        ref = deepcopy(model)
        ws = ModelWorkspace(model, MWOptions; kw...)
        refresult = optimize!(ref, MWOptions; kw...)
        for _ in 1:2
            copyto!(model.variables, deepcopy(start))
            result = optimize!(model, MWOptions; kw..., workspace=ws)
            @test all(k -> isapprox(landmarkvec(model.variables[k]), landmarkvec(ref.variables[k]); rtol=1e-12), eachindex(ref.variables))
            @test result.bestcost ≈ refresult.bestcost rtol=1e-10 atol=1e-20
        end
    end
end
