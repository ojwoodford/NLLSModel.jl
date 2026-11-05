using Test

@testset "NLLSModels Tests" begin
    include("CostStruct.jl")
    include("robust.jl")
    include("dynamicvars.jl")
    include("nonsquaredcost.jl")
    include("adaptivecost.jl")
    include("multithreading.jl")
    include("optimizeba.jl")
end
