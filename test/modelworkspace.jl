using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems

include("fixtures/loader.jl")
fixture("modelworkspace")

# ModelWorkspace (NLLSsolver D5): optimize!(model, options; workspace) with a ModelWorkspace reuses the model's setup as
# well as the optimizer's state, while the model's structure and the keywords are unchanged.

# The models of the table: each with the keywords it is solved with
const MW_CASES = (
    ("single static", () -> singleproblem(SVector{3, Float64}, 3, 1)[1], (numthreads=static(1),)),
    ("single static, default threads", () -> singleproblem(SVector{3, Float64}, 3, 1)[1], (;)),
    ("single Vector", () -> singleproblem(Vector{Float64}, 17, 1)[1], (numthreads=static(1),)),
    ("dense", () -> staticchain(5), (numthreads=static(1),)),
    ("sparse", () -> staticchain(30), (numthreads=static(1),)),
    ("subset", () -> staticchain(5), (unfixed=[2, 3, 4], numthreads=static(1))),
    ("one of several", () -> staticchain(5), (unfixed=3, numthreads=static(1))),
    ("by type", () -> mixedproblem()[1], (unfixed=SVector{2, Float64}, numthreads=static(1))),
    ("Schur, dense", () -> (Random.seed!(3); optlandmark_problem(4, 20)), (numthreads=static(1), storage=:dense)),
    ("Schur, sparse", () -> (Random.seed!(3); optlandmark_problem(4, 20)), (numthreads=static(1), storage=:sparse)),
    ("Float32", () -> staticchain(5), (numthreads=static(1), precision=Float32)),
    ("not residuals", quartmodel, (numthreads=static(1),)),
    ("Jacobian, single", () -> singleproblem(SVector{3, Float64}, 3, 1)[1], (numthreads=static(1), formulation=:jacobian)),
    ("Jacobian, dense", () -> staticchain(5), (numthreads=static(1), formulation=:jacobian)),
    ("Jacobian, subset", () -> staticchain(5), (unfixed=[2, 3, 4], numthreads=static(1), formulation=:jacobian)),
    ("Jacobian, eliminated", () -> (Random.seed!(3); optlandmark_problem(4, 20)), (numthreads=static(1), formulation=:jacobian)),
)

resetvars!(model, start) = copyto!(model.variables, deepcopy(start))

@testset "modelworkspace.jl" begin
    # Check the storage the sparse case is meant to exercise
    @test optimvars(staticchain(30); numthreads=static(1))[4] isa SparseMatrixCSC

    @testset "the same solution as without a workspace: $name, $(typed ? "typed" : "untyped")" for (name, make, kw) in MW_CASES, typed in (true, false)
        model = make()
        start = deepcopy(model.variables)
        ref = deepcopy(model)
        ws = typed ? ModelWorkspace(model, MWOptions; kw...) : ModelWorkspace()
        @test isempty(ws) == !typed
        result = optimize!(model, MWOptions; kw..., workspace=ws)
        refresult = optimize!(ref, MWOptions; kw...)
        @test model.variables == ref.variables
        @test result.bestcost == refresult.bestcost
        @test result.niterations == refresult.niterations
        @test !isempty(ws)
        # Again from the start: the stored setup is used, and starts from the variables' values at the call
        setup = ws.setup
        resetvars!(model, start)
        again = optimize!(model, MWOptions; kw..., workspace=ws)
        @test ws.setup === setup
        @test model.variables == ref.variables
        @test again.bestcost == refresult.bestcost
    end

    # A repeated call with a typed workspace allocates nothing (rules.md 21). The variables are static: a `Vector` variable
    # allocates in its own `update`.
    @testset "a repeated call allocates nothing: $name" for (name, make, kw) in MW_CASES
        name in ("single static, default threads", "single Vector", "by type", "Schur, sparse", "sparse", "not residuals") && continue
        model = make()
        ws = ModelWorkspace(model, MWOptions; kw...)
        @test mw_allocs(model, MWOptions, ws; kw...) == 0
    end

    @testset "a new setup when: $what" for (what, change!) in (
            ("unfixed changes", (m, kw) -> (m, merge(kw, (unfixed=3,)))),
            ("formulation changes", (m, kw) -> (m, merge(kw, (formulation=:jacobian,)))),
            ("precision changes", (m, kw) -> (m, merge(kw, (precision=Float32,)))),
            ("numthreads changes", (m, kw) -> (m, merge(kw, (numthreads=2,)))),
            ("a variable and a cost are added", (m, kw) -> (addcost!(m, Tgt(SVector(0.1, 0.2), addvariable!(m, SVector(1.0, 1.0)))); (m, kw))),
            ("a cost is added", (m, kw) -> (addcost!(m, SLnk(SVector(0.5, 0.5), 1, 5)); (m, kw))),
            ("the variable vector is replaced", (m, kw) -> (m.variables = copy(m.variables); (m, kw))),
            ("another model", (m, kw) -> (deepcopy(m), kw)))
        model = staticchain(5)
        kw = (; numthreads=static(1))
        ws = ModelWorkspace()
        optimize!(model, MWOptions; kw..., workspace=ws)
        setup = ws.setup
        model, kw = change!(model, kw)
        ref = deepcopy(model)
        result = optimize!(model, MWOptions; kw..., workspace=ws)
        @test ws.setup !== setup
        @test ws.key[1] === model
        refresult = optimize!(ref, MWOptions; kw...)
        @test model.variables == ref.variables
        @test result.bestcost == refresult.bestcost
    end

    @testset "a new setup when the caller's unfixed array changes" begin
        model = staticchain(5)
        unfixed = [2, 3]
        ws = ModelWorkspace()
        optimize!(model, MWOptions; unfixed, numthreads=static(1), workspace=ws)
        setup = ws.setup
        unfixed[2] = 4
        ref = deepcopy(model)
        optimize!(model, MWOptions; unfixed, numthreads=static(1), workspace=ws)
        @test ws.setup !== setup
        optimize!(ref, MWOptions; unfixed=[2, 4], numthreads=static(1))
        @test model.variables == ref.variables
    end

    # A variable of another size, and the cost on it: without a new setup the system would have the old size
    @testset "a new setup when a variable's size changes" begin
        model = singleproblem(Vector{Float64}, 3, 1)[1]
        ws = ModelWorkspace()
        optimize!(model, MWOptions; workspace=ws)
        setup = ws.setup
        model.variables[1] = [1.0, 2.0, 3.0, 4.0]
        get(model.costs, Tgt{Vector{Float64}})[1] = Tgt([0.5, 0.4, 0.3, 0.2], 1)
        optimize!(model, MWOptions; workspace=ws)
        @test ws.setup !== setup
        @test model.variables[1] ≈ [0.5, 0.4, 0.3, 0.2] atol=1e-6
    end

    # Values are not part of the key: a cost's data can change, and the next call uses it
    @testset "the same setup when a cost's data changes" begin
        model = staticchain(5)
        ws = ModelWorkspace(model, MWOptions; numthreads=static(1))
        optimize!(model, MWOptions; numthreads=static(1), workspace=ws)
        setup = ws.setup
        get(model.costs, Tgt{SVector{2, Float64}})[2] = Tgt(SVector(3.0, -3.0), 2)
        ref = deepcopy(model)
        optimize!(model, MWOptions; numthreads=static(1), workspace=ws)
        @test ws.setup === setup
        optimize!(ref, MWOptions; numthreads=static(1))
        @test model.variables == ref.variables
    end

    @testset "a typed workspace that can't hold a call is emptied" begin
        model = staticchain(5)
        ws = ModelWorkspace(model, MWOptions; unfixed=2, numthreads=static(1))
        ref = deepcopy(model)
        result = @test_logs (:warn, r"ModelWorkspace.*can't hold") match_mode=:any optimize!(model, MWOptions; numthreads=static(1), workspace=ws)
        @test isempty(ws)
        refresult = optimize!(ref, MWOptions; numthreads=static(1))
        @test model.variables == ref.variables
        @test result.bestcost == refresult.bestcost
    end

    @testset "empty! and isempty" begin
        model = staticchain(5)
        ws = ModelWorkspace(model, MWOptions; numthreads=static(1))
        @test !isempty(ws) && !isempty(ws.optimizer)
        @test empty!(ws) === ws
        @test isempty(ws) && isempty(ws.optimizer) && isnothing(ws.key)
        optimize!(model, MWOptions; numthreads=static(1), workspace=ws)
        @test !isempty(ws)
    end

    # optimize solves a copy of the model, so its setup is the copy's
    @testset "optimize (on a copy) is not reused" begin
        model = staticchain(5)
        ws = ModelWorkspace()
        copied, result = optimize(model, MWOptions; numthreads=static(1), workspace=ws)
        @test ws.key[1] === copied
        @test NLLSModel.cost(copied) == result.bestcost
        copied2, _ = optimize(model, MWOptions; numthreads=static(1), workspace=ws)
        @test ws.key[1] === copied2
        @test copied2.variables == copied.variables
    end

    @testset "errors" begin
        model = quartmodel()
        @test_throws ArgumentError ModelWorkspace(model, MWOptions; formulation=:normal)
        @test_throws ArgumentError ModelWorkspace(model, MWOptions; reduction=:x)
        @test_throws "formulation = :jacobian:" ModelWorkspace(model, MWOptions; formulation=:jacobian)
        @test_throws "formulation = :jacobian:" optimize!(model, MWOptions; formulation=:jacobian, workspace=ModelWorkspace())
    end
end
