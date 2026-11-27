using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization

include("fixtures/loader.jl")
fixture("optimizesingles")

# optimizesingles! (NLLSsolver D6): each named variable optimized on its own, the others fixed, with one linear system for
# each run of variables of one type and size. It gives what a loop over the variables through the front door gives.

@testset "optimizesingles.jl" begin
    @testset "as a loop through the front door: $name" for (name, make, variables, order) in (
            ("landmarks by type", () -> baproblem(), SVector{3, Float64}, nothing),
            ("cameras by index", () -> baproblem(), [5, 1, 3], [5, 1, 3]),
            ("mixed types and sizes", () -> mixedproblem()[1], 1:7, [3, 7, 2, 5, 6, 1, 4]))
        model = make()
        ref = deepcopy(model)
        order = something(order, findall(v -> v isa SVector{3, Float64}, model.variables))
        result = optimizesingles!(model, SingleOptions, variables)
        refresult = singlesloop!(ref, SingleOptions, order)
        @test model.variables == ref.variables
        @test result.startcost == refresult.startcost
        @test result.bestcost == refresult.bestcost
        @test result.niterations == refresult.niterations
        @test result.termination == refresult.termination
        @test result.bestcost < result.startcost
        # The model keeps its costs
        @test NLLSModel.countcosts(NLLSModel.costnum, model.costs) == NLLSModel.countcosts(NLLSModel.costnum, ref.costs)
    end

    @testset "precision = Float32" begin
        model = baproblem()
        ref = deepcopy(model)
        optimizesingles!(model, SingleOptions, [2, 4]; precision=Float32)
        singlesloop!(ref, SingleOptions, [2, 4]; precision=Float32)
        @test model.variables == ref.variables
    end

    @testset "no variable to optimize" for variables in (SVector{2, Float64}, Int[])
        model = baproblem()
        ref = deepcopy(model)
        result = optimizesingles!(model, SingleOptions, variables)
        @test model.variables == ref.variables
        @test result.niterations == 0 && result.startcost == 0 && result.bestcost == 0 && result.termination == 0
        @test result.iterator === :levenbergmarquardt
    end

    @testset "a variable without costs" begin
        model = baproblem()
        i = addvariable!(model, SVector(1.0, 2.0, 3.0))
        result = optimizesingles!(model, SingleOptions, [i])
        @test model.variables[i] == SVector(1.0, 2.0, 3.0)
        @test result.bestcost == 0
    end

    # After the first variable of a run, a variable costs no allocation: the run's linear system and the optimizer's
    # workspace are reused, and selecting its costs reuses the subproblem's storage
    @testset "a run allocates nothing per variable" begin
        model = baproblem()
        landmarks = findall(v -> v isa SVector{3, Float64}, model.variables)
        @test singlesrun_allocs(model, SingleOptions, landmarks) == 0
    end

    @testset "errors" begin
        model = baproblem()
        n = length(model.variables)
        @test_throws "between 1 and the number of variables" optimizesingles!(model, SingleOptions, [0])
        @test_throws "between 1 and the number of variables" optimizesingles!(model, SingleOptions, [n + 1])
        @test_throws "named once" optimizesingles!(model, SingleOptions, [2, 2])
        @test_throws ArgumentError optimizesingles!(model, SingleOptions, [2]; precision=Float16)
        @test_throws ArgumentError optimizesingles!(model, SingleOptions, [2]; max_dense_bytes=8)
    end
end
