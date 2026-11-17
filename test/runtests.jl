using Test
using OptimizationVariableInterface: nvars, update, update!

# Allocation checks (rules.md 21). `allocs_warm(f, args...)`: the bytes of one call of `f(args...)` after two warm calls,
# measured in a function barrier so that `@allocated` sees concrete types. On Julia 1.10 a call on a value of
# non-inferred type (a destructured `optimvars` result, a testset local) boxes, and `@allocated` would count that too.
allocs_warm(f::F, args::Vararg{Any, N}) where {F, N} = (f(args...); f(args...); @allocated f(args...))
# rules.md 21's 1.10 exception, as BlockLinearSystems' test/runtests.jl states it (the same name and bound): a threaded call
# may allocate up to 2 048 bytes per thread on 1.10, nothing on later versions. `threaded`: whether the call took the
# threaded path.
threaded_allocs_ok(bytes, threaded=Threads.nthreads() > 1) =
    bytes == 0 || (VERSION < v"1.11" && threaded && bytes <= 2048 * Threads.nthreads())

# Each test file is included through `timed_include`, which logs its wall time and compile time in seconds
# (`compile_time` is reported by `@timed` from Julia 1.11; NaN before) and keeps them for a table at the end.
const FILE_TIMES = Tuple{String, Float64, Float64}[]
function timed_include(f)
    s = @timed include(f)
    compile = hasproperty(s, :compile_time) ? s.compile_time : NaN
    push!(FILE_TIMES, (f, s.time, compile))
    @info f time = round(s.time; digits = 1) compile = round(compile; digits = 1)
    return nothing
end

function print_file_times()
    println("\nSeconds per test file (Julia ", VERSION, ", ", Threads.nthreads(), " thread(s)): total, compile")
    for (f, t, c) in FILE_TIMES
        println(rpad(f, 40), lpad(round(t; digits = 1), 8), lpad(round(c; digits = 1), 8))
    end
    println(rpad("total", 40), lpad(round(sum(x -> x[2], FILE_TIMES; init = 0.0); digits = 1), 8),
            lpad(round(sum(x -> x[3], FILE_TIMES; init = 0.0); digits = 1), 8))
end

# The table is printed even when a test fails
try
@testset "NLLSModel Tests" verbose = true begin
    timed_include("imports.jl")
    timed_include("publicapi.jl")
    timed_include("CostStruct.jl")
    timed_include("robust.jl")
    timed_include("dynamicvars.jl")
    timed_include("autodiff.jl")
    timed_include("nonsquaredcost.jl")
    timed_include("adaptivecost.jl")
    timed_include("multithreading.jl")
    timed_include("gramkernels.jl")
    timed_include("problemapi.jl")
    timed_include("runtimeblocks.jl")
    timed_include("bigvariable.jl")
    timed_include("optimizeba.jl")
    timed_include("optimization.jl")
    timed_include("frontdoor.jl")
    timed_include("tallproblems.jl")
    timed_include("tallkernels.jl")
    timed_include("tallformulation.jl")
    timed_include("manyvariables.jl")
    timed_include("tallcontract.jl")
    timed_include("talliterators.jl")
    timed_include("tallalloc.jl")
    timed_include("tallelimination.jl")
    timed_include("eliminablesizes.jl")
    timed_include("tallcombinations.jl")
    timed_include("NLPModel.jl")
    timed_include("nls_interface.jl")
    timed_include("keywords.jl")
end
finally
    print_file_times()
end
