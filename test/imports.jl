using Test

include("fixtures/loader.jl")
fixture("imports")

@testset "Imports" begin
    root = dirname(@__DIR__)
    srcfiles = filter(endswith(".jl"), readdir(joinpath(root, "src"); join=true))
    @test isempty(ImportCheck.unused_imports(srcfiles))
    # An extension is its file and the files that includes: a name its file imports may be used in any of them, and
    # only its file imports (rules.md 44). A file another one includes is not an extension of its own.
    extfiles = filter(endswith(".jl"), readdir(joinpath(root, "ext"); join=true))
    extmodules = Dict(file => ImportCheck.modulefiles(file) for file in extfiles)
    included = Set(f for files in values(extmodules) for f in files[2:end])
    @testset "ext/$(basename(file))" for file in filter(!in(included), extfiles)
        @test isempty(ImportCheck.unused_imports(extmodules[file]))
        @test all(f -> isempty(ImportCheck.importlines(f)), extmodules[file][2:end])
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
    # The check follows includes: a name used only in an included file is used, one used in neither is not
    mktempdir() do dir
        file = joinpath(dir, "m.jl")
        write(file, "module M\nusing A: a, b\ninclude(\"part.jl\")\nend\n")
        mkdir(joinpath(dir, "sub"))
        write(joinpath(dir, "part.jl"), "f(x) = a(x)\ninclude(\"sub/more.jl\")\n")
        write(joinpath(dir, "sub", "more.jl"), "g(x) = x\n")
        files = ImportCheck.modulefiles(file)
        @test files == [file, joinpath(dir, "part.jl"), joinpath(dir, "sub", "more.jl")]
        @test ImportCheck.unused_imports([file]) == [:a, :b]
        @test ImportCheck.unused_imports(files) == [:b]
    end
    # The check sees an unused name, and does not count a qualified use
    mktempdir() do dir
        file = joinpath(dir, "m.jl")
        write(file, "import A: a, b, c as d\nusing B: e\nf(x) = a(x) + A.b(x) + d + e\n")
        @test ImportCheck.unused_imports([file]) == [:b]
    end
end
