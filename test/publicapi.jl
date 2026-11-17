using NLLSModel, Test

include("fixtures/loader.jl")
fixture("publicapi")

@testset "Public API" begin
    @test sort(names(NLLSModel)) == PUBLIC_NAMES
    @test length(PUBLIC_NAMES) == 42
    @testset "docstring: $name" for name in PUBLIC_NAMES
        @test hasdocstring(NLLSModel, name)
    end
    @test apireference_names(joinpath(dirname(@__DIR__), "docs", "src", "api.md")) == PUBLIC_NAMES
    @testset "removed: $name" for name in REMOVED_NAMES
        @test !isdefined(NLLSModel, name)
    end
    @testset "internal: $name" for name in INTERNAL_NAMES
        @test isdefined(NLLSModel, name)
        @test name ∉ names(NLLSModel)
    end
end
