using Test
using OptimizationVariableInterface: nvars, update, update!

@testset "NLLSModel Tests" begin
    include("CostStruct.jl")
    include("robust.jl")
    include("dynamicvars.jl")
    include("nonsquaredcost.jl")
    include("adaptivecost.jl")
    include("multithreading.jl")
    include("gramkernels.jl")
    include("problemapi.jl")
    include("runtimeblocks.jl")
    include("bigvariable.jl")
    include("optimizeba.jl")
    include("optimization.jl")
    include("frontdoor.jl")
    include("tallproblems.jl")
    include("tallkernels.jl")
    include("tallformulation.jl")
    include("tallcontract.jl")
    include("talliterators.jl")
    include("tallalloc.jl")
    include("NLPModel.jl")
    include("nls_interface.jl")
end
