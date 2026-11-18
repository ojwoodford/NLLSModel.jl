using Test

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

# Test groups, chosen by name in ARGS (`Pkg.test(test_args = ["serial"])`, or ARGS set before including this file); all
# of them when ARGS is empty. `threaded`: the tests whose behaviour depends on the thread count, the only ones that need
# running on more than one thread. `integration`: the third-party solvers. The full check before a push: all groups on
# one thread, then `threaded` on four.
const TEST_GROUPS = (
    serial = ["imports.jl", "publicapi.jl", "CostStruct.jl", "robust.jl", "dynamicvars.jl", "autodiff.jl", "nonsquaredcost.jl",
        "adaptivecost.jl", "gramkernels.jl", "problemapi.jl", "runtimeblocks.jl", "bigvariable.jl", "optimizeba.jl",
        "optimization.jl", "frontdoor.jl", "tallkernels.jl", "tallformulation.jl", "tallcontract.jl", "talliterators.jl",
        "tallalloc.jl", "tallelimination.jl", "eliminablesizes.jl", "tallcombinations.jl", "NLPModel.jl", "nls_interface.jl",
        "keywords.jl", "recommender.jl"],
    threaded = ["multithreading.jl", "gramkernels_threaded.jl", "runtimeblocks_threaded.jl", "optimization_threaded.jl",
        "frontdoor_threaded.jl", "tallformulation_threaded.jl", "manyvariables.jl", "tallalloc_threaded.jl",
        "tallelimination_threaded.jl", "eliminablesizes_threaded.jl", "nls_interface_threaded.jl", "keywords_threaded.jl"],
    integration = ["nls_integration.jl"],
)
const SELECTED_GROUPS = isempty(ARGS) ? collect(keys(TEST_GROUPS)) : Symbol.(ARGS)
for g in SELECTED_GROUPS
    haskey(TEST_GROUPS, g) || error("unknown test group $g; the groups are $(join(keys(TEST_GROUPS), ", "))")
end

# The table is printed even when a test fails
try
@testset "NLLSModel Tests" verbose = true begin
    for g in SELECTED_GROUPS, f in TEST_GROUPS[g]
        timed_include(f)
    end
end
finally
    print_file_times()
end
