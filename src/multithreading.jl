# Threading decisions. Each kind of work has its own estimate, the count of numbers it writes, summed over all costs, with
# per cost `m` the residual length (1 for a general `AbstractCost`) and `n` the total dimension of its unfixed variables:
#   cost evaluation                 m              (residual entries)                         per-thread floor `min_work_cost`
#   Jacobian fill, r and J              m * (n + 1)    (Jacobian entries and the residual)        per-thread floor `min_work`
#   Hessian fill, JᵀJ and Jᵀr        m * n * (n + 1) (Jacobian entries times columns, Gram)    per-thread floor `min_work`
# A general `AbstractCost` has a Hessian, so it counts as `m = 1` in the Hessian-fill form. The work of the residual
# function itself is not estimated. The estimates are computed once, at setup (`workestimates`), and each kind of work
# is split over `threadcount(work, floor, numthreads, costsplit)` threads, decided then too: never more than the largest
# number of costs of one type (`costsplit`), and for the Hessian fill at most `MAX_THREADS_HESSIAN` (`hessianthreads`).
#
# The defaults were measured on an Apple M1 Pro (8 performance and 2 efficiency cores), Julia 1.13.3, -t 8, 2026-11-14/15,
# three runs. For each family and size every kind of work was timed on exactly 1 to 8 threads (minimum over repeated
# evaluations), and the count the rule chooses for T = 2, 4 and 8 was compared with serial and with the best count up to
# T. Worst cases over the three runs, for the families with cheap residuals (static variables, hand-sized residual code),
# at the defaults:
#
#   kind          floor             bundle adjustment (2x12)          pose chain (6x12)
#   cost          120               1.04x serial, 1.58-1.90x best     1.00-1.23x serial, 1.29-2.06x best
#   Jacobian fill     280               1.00x serial, 1.36-1.95x best     1.00-1.11x serial, 1.47-1.77x best
#   Hessian fill   4000, at most 4   1.00x serial, 1.19-1.21x best     1.00x serial, 1.12-1.15x best
#
# The cost and Jacobian floors are compromises: no per-thread floor keeps every size within 25% of the best count, because the
# best count jumps between sizes near each switch point and the best floor moved between runs (Jacobian: 130 to 278). The
# Hessian fill is capped because each thread fills and then sums its own copy of the system, work in proportion to the size
# of the system: without the cap, on 8 threads, 4 to 7 threads beat 8 by up to 2.3x from an estimate of about 60 000. The
# cap of 4 is a starting value for this machine; more cores, or a costly residual, may want more.
#
# The costly residuals (Procrustes) and the dynamic-variable autodiff (curve fit) would thread profitably well below these
# floors, because the per-cost work of evaluating the residual and its derivatives is not in the estimates; at these
# floors they thread later than they could, up to 4 to 6x slower than their best count at small sizes (decided
# 2026-11-14: no per-cost term). Before the count was bounded by the number of costs of one type, a model of a single
# costly cost was given up to 4 threads for the Hessian fill (Procrustes, one cost: up to 1.39x slower than serial).
const MIN_WORK_PER_THREAD_COST = 120
const MIN_WORK_PER_THREAD_JACOBIAN = 280
const MIN_WORK_PER_THREAD_HESSIAN = 4_000
const MAX_THREADS_HESSIAN = 4

# The three estimates for the variables selected by `unfixed` (an index, or a mask over `problem.variables`)
struct WorkEstimates
    cost::Int
    jacobianfill::Int
    hessianfill::Int
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
    w_cost = w_jac = w_lin = 0
    for c in vec
        m = costrows(c)
        vinds = varindices(c)
        n = 0
        for v in (vinds isa Integer ? (vinds,) : vinds)
            n += sizes[v]
        end
        w_cost += m
        w_jac += m * (n + 1)
        w_lin += m * n * (n + 1)
    end
    return (w_cost, w_jac, w_lin)
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

# The most threads an evaluation can keep busy: every evaluation splits the costs of each type into one chunk per thread,
# so a thread beyond the largest number of costs of one type is given nothing (and on the Hessian path would only zero and
# sum its copy of the system)
costsplit(costs::CostStruct) = maximum(length, values(costs); init=0)

# The one threading rule, for every kind of work: the number of threads for an estimate `work` at a per-thread floor
# `floor`. One thread below twice the floor, `work ÷ floor` threads above it, and never more than `numthreads` or than
# `costsplit`, the largest number of costs of one type; a floor of 0 uses every thread there is work for. Decided once,
# at setup, from the estimates of `workestimates`.
function threadcount(work::Integer, floor::Integer, numthreads::Integer, costsplit::Integer)
    maxthreads = min(Int(numthreads), Int(costsplit))
    maxthreads <= 1 && return 1
    floor <= 0 && return maxthreads
    work < 2 * floor && return 1
    return Int(clamp(work ÷ floor, 1, maxthreads))
end

# The Hessian fill: the same rule, at most `MAX_THREADS_HESSIAN` threads (each thread fills its own copy of the system)
hessianthreads(work::Integer, floor::Integer, numthreads::Integer, costsplit::Integer) =
    min(threadcount(work, floor, numthreads, costsplit), MAX_THREADS_HESSIAN)

# The cost, summed over `nthreads` chunks of the costs, one per thread
function cost_multithreaded(vars, costs::CostStruct, nthreads::Int)::Float64
    if nthreads <= 1
        return sum(bindleadingargs(computecost, vars), costs; init=0.0)
    end
    tot = 0.0
    args = threadargs((vars, costs))
    GC.@preserve vars costs begin
        @batch reduction=((+, tot),) for taskid in 1:nthreads
            vars_, costs_ = fromthreadargs(args)
            tot += sumsubset(bindleadingargs(computecost, vars_), costs_, bindleadingargs(ithchunk, nthreads, taskid); init=0.0)
        end
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

    args = threadargs((linsystem, vars, costs))
    GC.@preserve linsystem vars costs begin
        # Phase 1: Parallel gradient & cost chunk evaluation across threads
        @batch for taskid in 1:N
            linsystem_, vars_, costs_ = fromthreadargs(args)
            ls = LinearSystem(linsystem_, taskid)
            zero!(ls)
            linsystem_.costs[taskid] = costgradhess_chunk!(ls, vars_, costs_, N, taskid)
        end

        # Phase 2: Parallel tree reduction across the N buffers
        n = N
        while n > 1
            n_ = (n + 1) >> 1
            m = n - n_
            @batch for taskid in 1:m
                linsystem_ = fromthreadargs(args)[1]
                add!(linsystem_.ls[taskid], linsystem_.ls[taskid + n_])
                linsystem_.costs[taskid] += linsystem_.costs[taskid + n_]
            end
            n = n_
        end
    end

    return linsystem.costs[1]
end
