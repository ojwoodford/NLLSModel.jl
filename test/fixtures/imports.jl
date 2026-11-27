using Test

# Every name in an `import M: a, b` or `using M: a, b` list of the package and of its extensions is referenced in the
# module that imports it (rules.md 11: no unused imports). A reference is a bare use of the name outside import and
# using statements; `M.a` does not count, since it needs no import. The package's `src/` files form one module; each
# extension in `ext/` is a module of its own, made of its file and the files that includes (`modulefiles`).
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

# rules.md 44: the lines of the `using` and `import` statements in `ex`, not counting those of a submodule it defines
function importlines(ex, lines=Int[], line=0)
    ex isa Expr || return lines
    ex.head === :module && return lines
    if ex.head === :import || ex.head === :using
        push!(lines, line)
        return lines
    end
    for a in ex.args
        a isa LineNumberNode ? (line = a.line) : importlines(a, lines, line)
    end
    return lines
end
importlines(file::AbstractString) = importlines(Meta.parseall(read(file, String); filename=file))

# The body of the first module defined in `ex` (under its docstring, if it has one)
function modulebody(ex)
    ex isa Expr || return nothing
    ex.head === :module && return ex.args[3]
    for a in ex.args
        body = modulebody(a)
        body === nothing || return body
    end
    return nothing
end

# The module file's imports, and the line of its first top-level `include`
function moduleimports(file)
    body = modulebody(Meta.parseall(read(file, String); filename=file))
    line, first_include = 0, typemax(Int)
    for a in body.args
        a isa LineNumberNode && (line = a.line)
        if a isa Expr && a.head === :call && a.args[1] === :include
            first_include = line
            break
        end
    end
    return importlines(body), first_include
end

# The paths of the `include("…")` calls in `ex` (literal paths only), relative to `dir`
function includes!(paths, ex, dir)
    ex isa Expr || return paths
    if ex.head === :call && length(ex.args) == 2 && ex.args[1] === :include && ex.args[2] isa AbstractString
        push!(paths, normpath(joinpath(dir, ex.args[2])))
    else
        foreach(a -> includes!(paths, a, dir), ex.args)
    end
    return paths
end

# The files of the module that `file` defines: `file`, then every file it includes, directly or through another
function modulefiles(file)
    files = [normpath(file)]
    i = 1
    while i <= length(files)
        f = files[i]
        for inc in includes!(String[], Meta.parseall(read(f, String); filename=f), dirname(f))
            inc in files || push!(files, inc)
        end
        i += 1
    end
    return files
end

end # module ImportCheck
