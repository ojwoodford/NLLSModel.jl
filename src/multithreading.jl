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
#
# The defaults are the break-evens of the families with cheap residuals (static variables, hand-sized residual code), on
# 4 threads, measured on an Apple M1 Pro (8 performance and 2 efficiency cores), Julia 1.13.3, 2026-11-14, by
# `benchmark/threading.jl`. Break-even as (lo, hi]: lo the largest estimate at which the threaded evaluation was not
# faster, hi the next size measured.
#
#   kind           threads  bundle adjustment (2x12)  pose chain (6x12)   curve fit, dynamic (1x5)  Procrustes d=5/10/20
#   cost             2       (84, 136]                 (132, 216]          (52, 84]*                 (30,50] (30,50] (0,20]
#                    4       (216, 346]                (132, 216]          (52, 84]                  (0,20] each
#                    8       (346, 554]                (348, 552]          (52, 84]                  (30,50] (0,20] (0,20]
#   tall fill        2       (546, 884]                (234, 312]          (0, 18]                   (0,55] (0,460] (0,3820]
#                    4       (338, 546]                (546, 858]          (0, 18]                   same
#                    8       (338, 546]                (546, 858]          (0, 18]                   same
#   linearization    2       (2184, 3744]              (5616, 9360]        (0, 30]                   (0,550] (0,20700] (0,725800]
#                    4       (9360, 14976]             (9360, 14976]       (60, 90]                  (1100,1650] (0,20700] (0,725800]
#                    8       (14976, 23712]            (38376, 60840]      (120, 210]                (1100,1650] (0,20700] (0,725800]
#   * on 2 threads two sizes measured 0.64x between 1.98x and 1.92x (an outlier, probably GC); about (52, 84] without them.
#
# No single floor per kind of work is within a factor of two of every family: the costly residuals (Procrustes) and the
# dynamic-variable autodiff (curve fit) thread profitably at two costs, 7x (tall) to 45x (linearization) below the cheap
# families, because the per-cost work of evaluating the residual and its derivatives is not in the estimates. The floors
# are set at the cheap families, so a costly residual threads later than it could, never too early (decided 2026-11-14).
# On 8 threads the pose chain's linearization pays only from about 38 000 to 61 000, above this floor.
const MIN_WORK_FOR_THREADED_COST = 350
const MIN_WORK_FOR_THREADED_LINEARIZATION = 15_000
const MIN_WORK_PER_THREAD = 450 # tall fill: threaded from 2 * MIN_WORK_PER_THREAD = 900, one thread per MIN_WORK_PER_THREAD

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
