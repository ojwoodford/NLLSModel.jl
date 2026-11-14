using Polyester

# Threading decisions. Each kind of work has its own estimate, the count of numbers it writes, summed over all costs, with
# per cost `m` the residual length (1 for a general `AbstractCost`) and `n` the total dimension of its unfixed variables:
#   cost evaluation                 m              (residual entries)                        floor `min_work_cost`
#   tall residual and Jacobian fill m * (n + 1)    (Jacobian entries and the residual)       floor 2 * `min_work_per_thread`
#   normal-equations linearization  m * n * (n + 1) (Jacobian entries times columns, Gram)   floor `min_work`
# A general `AbstractCost` has a Hessian, so it counts as `m = 1` in the linearization form. The work of the residual
# function itself is not estimated: the floors are calibrated on cheap residuals, so a costly one threads later than it
# could, never earlier than it should. The estimates are computed once, at setup (`workestimates`); an evaluation compares
# an integer. Problems with less work than the floor are evaluated on a single thread.
# The defaults are the values from the old estimate (residual length times number of variables squared), not yet
# recalibrated in these units.
const MIN_WORK_FOR_THREADED_COST = 15_000
const MIN_WORK_FOR_THREADED_LINEARIZATION = 10_000

# The three estimates for the variables selected by `unfixed` (an index, or a mask over `problem.variables`)
struct WorkEstimates
    cost::Int
    tall::Int
    linearization::Int
end

unfixedsizes(problem, unfixed::Integer) = [i == unfixed ? Int(dynamic(nvars(v))) : 0 for (i, v) in enumerate(problem.variables)]
unfixedsizes(problem, unfixed::AbstractVector{Bool}) = [unfixed[i] ? Int(dynamic(nvars(v))) : 0 for (i, v) in enumerate(problem.variables)]
unfixedsizes(problem, ::Nothing) = [Int(dynamic(nvars(v))) for v in problem.variables]
function unfixedsizes(problem, unfixed_inds::AbstractVector{<:Integer})
    sizes = zeros(Int, length(problem.variables))
    for i in unfixed_inds
        sizes[i] = Int(dynamic(nvars(problem.variables[i])))
    end
    return sizes
end

costrows(c::AbstractResidual) = Int(dynamic(nres(c)))
costrows(::AbstractCost) = 1

function vecwork(vec::Vector, sizes::Vector{Int})
    w_cost = w_tall = w_lin = 0
    for c in vec
        m = costrows(c)
        vinds = varindices(c)
        n = 0
        for v in (vinds isa Integer ? (vinds,) : vinds)
            n += sizes[v]
        end
        w_cost += m
        w_tall += m * (n + 1)
        w_lin += m * n * (n + 1)
    end
    return (w_cost, w_tall, w_lin)
end

function workestimates(problem::Model, unfixed)
    sizes = unfixedsizes(problem, unfixed)
    w = foldl((a, b) -> a .+ b, map(vec -> vecwork(vec, sizes), values(problem.costs)); init=(0, 0, 0))
    return WorkEstimates(w...)
end

# The cost estimate needs no variables, so a standalone `cost(model)` uses it too
costestimate(costs::CostStruct) = Int(countcosts(costvecrows, costs))
costvecrows(vec::Vector{C}) where C = C <: AbstractResidual ? Int(dynamic(resnum(vec))) : length(vec)

@inline function chunk_indices(len::Integer, nchunks::Integer, chunkid::Integer)
    chunkid > nchunks && return 1:0
    q, r = divrem(len, nchunks)
    start = (chunkid - 1) * q + min(chunkid - 1, r) + 1
    stop = start + q - 1 + (chunkid <= r ? 1 : 0)
    return start:stop
end

@inline function ithchunk(N, taskid, collection)
    return view(collection, chunk_indices(length(collection), N, taskid))
end

function cost_multithreaded(vars, costs::CostStruct, ::StaticInt{N}, threaded::Bool)::Float64 where {N}
    if !threaded
        return sum(bindleadingargs(computecost, vars), costs; init=0.0)
    end
    tot = 0.0
    @batch reduction=((+, tot),) for taskid in 1:N
        tot += sumsubset(bindleadingargs(computecost, vars), costs, bindleadingargs(ithchunk, N, taskid); init=0.0)
    end
    return tot
end

costgradhess_chunk!(ls, vars, costs::CostStruct, N, taskid) = sumsubset(bindleadingargs(costgradhess!, ls, vars), costs, bindleadingargs(ithchunk, N, taskid); init=0.0)

function costgradhess!(linsystem::LinearSystemShared{F, V, N}, vars::AbstractVector, costs::CostStruct)::Float64 where {F, V, N}
    if N == 1
        ls = LinearSystem(linsystem, 1)
        zero!(ls)
        linsystem.costs[1] = costgradhess!(ls, vars, costs)
        return linsystem.costs[1]
    end

    # Phase 1: Parallel gradient & cost chunk evaluation across threads
    @batch for taskid in 1:N
        ls = LinearSystem(linsystem, taskid)
        zero!(ls)
        linsystem.costs[taskid] = costgradhess_chunk!(ls, vars, costs, N, taskid)
    end

    # Phase 2: Parallel tree reduction across the N buffers
    n = N
    while n > 1
        n_ = (n + 1) >> 1
        m = n - n_
        @batch for taskid in 1:m
            add!(linsystem.ls[taskid], linsystem.ls[taskid + n_])
            linsystem.costs[taskid] += linsystem.costs[taskid + n_]
        end
        n = n_
    end

    return linsystem.costs[1]
end
