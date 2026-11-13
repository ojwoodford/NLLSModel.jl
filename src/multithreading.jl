using Polyester

# Problems with less work than these are evaluated on a single thread. Work is estimated by `costwork` (residual length
# times variables squared; one scalar residual of one variable is one unit). The defaults come from measurements of affine
# bundle adjustment problems on an Apple-silicon laptop with 4 threads (serial time per unit of work: about 0.7 ns for the
# cost, 17 ns for the linearization; threading the cost already wins at 16 000 units, 10.6 -> 3.2 us, while threading the
# linearization loses at 2 000 units, 32 -> 37 us, breaks even near 10 000 and gives 1.6x at 160 000). Both can be set per
# problem with the `min_work_cost` and `min_work` options of `optimvars`.
const MIN_WORK_FOR_THREADED_COST = 15_000 # Evaluating the cost alone: roughly 10 us of serial work
const MIN_WORK_FOR_THREADED_LINEARIZATION = 10_000 # Linearizing: the break-even measured above

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

function cost_multithreaded(vars, costs::CostStruct, ::StaticInt{N}, min_work::Integer=MIN_WORK_FOR_THREADED_COST)::Float64 where {N}
    if countcosts(costwork, costs) < min_work
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
