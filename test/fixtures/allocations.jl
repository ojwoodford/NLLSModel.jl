# Allocation checks (rules.md 21). `allocs_warm(f, args...)`: the bytes of one call of `f(args...)` after two warm calls,
# measured in a function barrier so that `@allocated` sees concrete types. On Julia 1.10 a call on a value of
# non-inferred type (a destructured `optimvars` result, a testset local) boxes, and `@allocated` would count that too.
allocs_warm(f::F, args::Vararg{Any, N}) where {F, N} = (f(args...); f(args...); @allocated f(args...))
# rules.md 21's 1.10 exception, as BlockLinearSystems' test/runtests.jl states it (the same name and bound): a threaded call
# may allocate up to 2 048 bytes per thread on 1.10, nothing on later versions. `threaded`: whether the call took the
# threaded path.
threaded_allocs_ok(bytes, threaded=Threads.nthreads() > 1) =
    bytes == 0 || (VERSION < v"1.11" && threaded && bytes <= 2048 * Threads.nthreads())

# Allocation of one call, after a warm-up call (in a function, so the measurement sees concrete types)
warmallocs_fdf(fdf!, g, H, x) = (fdf!(g, H, x); @allocated fdf!(g, H, x))
warmallocs_f(f, x) = (f(x); @allocated f(x))

# The allocations of one warm call of each evaluator (resjac!, res!, f). `optimvars_nls`'s result is not inferred, so
# each evaluator is measured in its own function barrier (`allocs_warm`, runtests.jl): on Julia 1.10, with all three in one
# function, inference of that function gives up and the calls become dynamic, which `@allocated` would count.
function warmallocs(model; numthreads=static(1), min_work=5000, storage=:dense, kwargs...)
    f, res!, resjac!, x0, J = optimvars_nls(model; numthreads, min_work, storage, kwargs...)
    return evalallocs(f, res!, resjac!, x0, J)
end

function evalallocs(f, res!, resjac!, x0, J)
    r = zeros(NLLSModel.nresiduals(res!.problem))
    return allocs_warm(resjac!, r, J, x0), allocs_warm(res!, r, x0), allocs_warm(f, x0)
end
