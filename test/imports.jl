using Test

# Every name in an `import M: a, b` or `using M: a, b` list of the package and of its extensions is referenced in the
# module that imports it (rules.md 11: no unused imports). A reference is a bare use of the name outside import and
# using statements; `M.a` does not count, since it needs no import. The package's `src/` files form one module; each
# file in `ext/` is a module of its own.
module ImportCheck

# Names listed after the colon of `import M: …` / `using M: …`
function listednames(ex::Expr)
    names = Symbol[]
    for arg in ex.args
        if arg isa Expr && arg.head === :(:)
            for item in arg.args[2:end]
                path = item.head === :as ? item.args[2] : item # `import M: a as b` binds b
                push!(names, path isa Symbol ? path : path.args[end])
            end
        end
    end
    return names
end

# Collect the listed imports and the bare symbols used anywhere else
function scan!(imported, used, ex)
    if ex isa Symbol
        push!(used, ex)
    elseif ex isa Expr
        if ex.head === :import || ex.head === :using
            append!(imported, listednames(ex))
        elseif ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
            scan!(imported, used, ex.args[1]) # `M.a`: only `M` is a reference
        else
            foreach(a -> scan!(imported, used, a), ex.args)
        end
    end
    return nothing
end

function unused_imports(files)
    imported, used = Symbol[], Set{Symbol}()
    for file in files
        scan!(imported, used, Meta.parseall(read(file, String); filename=file))
    end
    return sort(unique(filter(n -> n ∉ used, imported)))
end

end # module ImportCheck

@testset "Imports" begin
    root = dirname(@__DIR__)
    srcfiles = filter(endswith(".jl"), readdir(joinpath(root, "src"); join=true))
    @test isempty(ImportCheck.unused_imports(srcfiles))
    @testset "ext/$(basename(file))" for file in filter(endswith(".jl"), readdir(joinpath(root, "ext"); join=true))
        @test isempty(ImportCheck.unused_imports([file]))
    end
    # The check sees an unused name, and does not count a qualified use
    mktempdir() do dir
        file = joinpath(dir, "m.jl")
        write(file, "import A: a, b, c as d\nusing B: e\nf(x) = a(x) + A.b(x) + d + e\n")
        @test ImportCheck.unused_imports([file]) == [:b]
    end
end
