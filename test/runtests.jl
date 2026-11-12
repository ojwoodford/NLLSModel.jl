using Test
using OptimizationVariableInterface: nvars, update, update!

@testset "NLLSModel Tests" begin
    include("CostStruct.jl")
    include("robust.jl")
    include("dynamicvars.jl")
    include("nonsquaredcost.jl")
    include("adaptivecost.jl")
    include("multithreading.jl")
    include("problemapi.jl")
    include("runtimeblocks.jl")
    include("optimizeba.jl")
    include("optimization.jl")
    include("NLPModel.jl")
    include("nls_interface.jl")
end
