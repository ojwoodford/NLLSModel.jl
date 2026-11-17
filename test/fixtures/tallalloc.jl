const AllocExt = Base.get_extension(NLLSModel, :SecondOrderOptimizationExt)

function staticrobust(kernel; n=4)
    rng = MersenneTwister(5)
    m = Model(SVector{2, Float64}, TlRob{Tgt{SVector{2, Float64}}, typeof(kernel)})
    foreach(k -> addvariable!(m, randn(rng, SVector{2, Float64})), 1:n)
    foreach(k -> addcost!(m, TlRob(Tgt(randn(rng, SVector{2, Float64}), k), kernel)), 1:n)
    return m
end
staticplain(; n=4) = (rng = MersenneTwister(5); m = Model(SVector{2, Float64}, Tgt{SVector{2, Float64}});
    foreach(k -> addvariable!(m, randn(rng, SVector{2, Float64})), 1:n);
    foreach(k -> addcost!(m, Tgt(randn(rng, SVector{2, Float64}), k)), 1:n); m)

# A warm call with the static copies the optimizer's static system passes, with and without prezeroed
prezeroedcall(rj, r, J, x) = rj(r, J, x; prezeroed=true)
function staticallocs(rj, r, J, x)
    rj(r, J, x); prezeroedcall(rj, r, J, x)
    a = @allocated rj(r, J, x)
    b = @allocated prezeroedcall(rj, r, J, x)
    return a, b
end
# A warm solve of a single static variable (`optimize`, which returns the solution) through the static system. The
# workspace is built here and the solve measured in a barrier that takes it: on Julia 1.10 the workspace's type is
# inferred only as a union (static or dense system, chosen at run time), so a solve in the same function is a dynamic call.
staticsolve(x0, obj, opts, ws) = SecondOrderOptimization.optimize(x0, obj, opts; workspace=ws)
function staticsolve_allocs(f, rj, x0, J)
    opts = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=10, reldcost=0.0, absdcost=0.0, dstep=0.0, gtol=0.0, maxstall=10^6)
    obj = JacobianObjective(f, AllocExt.TallResJac(rj), J)
    ws = OptimizationWorkspace(x0, obj, opts)
    return allocs_warm(staticsolve, x0, obj, opts, ws)
end

# A warm whole solve through the optimizer (`optimize!`), measured in a barrier: the bytes, and the result
function solveallocs(x0, obj, opts, ws)
    SecondOrderOptimization.optimize!(x0, obj, opts; workspace=ws)
    a = @allocated res = SecondOrderOptimization.optimize!(x0, obj, opts; workspace=ws)
    return a, res
end
