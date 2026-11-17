using Test

include("fixtures/loader.jl")
fixture("imports")

@testset "Imports" begin
    root = dirname(@__DIR__)
    srcfiles = filter(endswith(".jl"), readdir(joinpath(root, "src"); join=true))
    @test isempty(ImportCheck.unused_imports(srcfiles))
    @testset "ext/$(basename(file))" for file in filter(endswith(".jl"), readdir(joinpath(root, "ext"); join=true))
        @test isempty(ImportCheck.unused_imports([file]))
    end
    # rules.md 44: imports once, at the top of the module file, before the include lines; none in an included file
    mainfile = joinpath(root, "src", "NLLSModel.jl")
    lines, first_include = ImportCheck.moduleimports(mainfile)
    @test !isempty(lines) && maximum(lines) < first_include < typemax(Int)
    @testset "src/$(basename(file)) has no import" for file in filter(!=(mainfile), srcfiles)
        @test isempty(ImportCheck.importlines(file))
    end
    # The check sees an import in a file, but not in a submodule or a docstring
    mktempdir() do dir
        file = joinpath(dir, "m.jl")
        write(file, "f(x) = x\n\"\"\"\nusing A\n\"\"\"\ng(x) = x\nmodule Sub\nusing B\nend\nimport C: c\n")
        @test ImportCheck.importlines(file) == [9]
    end
    # The check sees an unused name, and does not count a qualified use
    mktempdir() do dir
        file = joinpath(dir, "m.jl")
        write(file, "import A: a, b, c as d\nusing B: e\nf(x) = a(x) + A.b(x) + d + e\n")
        @test ImportCheck.unused_imports([file]) == [:b]
    end
end
