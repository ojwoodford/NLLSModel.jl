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

@testset "NLLSModel Tests" begin
    include("imports.jl")
    include("publicapi.jl")
    include("CostStruct.jl")
    include("robust.jl")
    include("dynamicvars.jl")
    include("autodiff.jl")
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
    include("tallelimination.jl")
    include("eliminablesizes.jl")
    include("tallcombinations.jl")
    include("NLPModel.jl")
    include("nls_interface.jl")
    include("keywords.jl")
end
