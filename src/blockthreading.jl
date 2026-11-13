using Polyester

# Threaded linearization of dense and sparse block systems without per-thread copies of the system.
#
# The blocks are split into contiguous ranges, one per thread. A thread processes every cost that touches a block in its
# range but only writes the blocks in its own columns (see `BlockRange`), so threads never write the same memory and
# nothing has to be summed afterwards. A cost whose variables belong to several threads is evaluated by each of them
# (the price of not copying the system); its value is counted once, by the thread owning its first unfixed block.

struct BlockThreadPlan
    veclengths::Vector{Int}
    ranges::Vector{BlockRange}       # Block columns written by each thread
    primary::Matrix{Vector{Int}}     # [cost vector, thread] -> costs whose value this thread counts
    secondary::Matrix{Vector{Int}}   # [cost vector, thread] -> other costs that touch this thread's blocks
    costs::Vector{Float64}           # Per-thread cost totals
end

plan_is_valid(plan::BlockThreadPlan, costs::CostStruct) = plan.veclengths == collect(map(length, values(costs)))

# Blocks (in order of the cost's variables) of the cost's unfixed variables
function unfixedblocks(cost, blockindices)
    vinds = varindices(cost)
    vinds isa Integer && return blockindices[vinds] == 0 ? Int[] : Int[blockindices[vinds]]
    return Int[Int(blockindices[v]) for v in vinds if blockindices[v] != 0]
end

function build_block_thread_plan(problem::Model, blockindices::AbstractVector, nblocks::Integer, N::Integer)
    vecs = values(problem.costs)
    nv = length(vecs)
    degree = zeros(Int, nblocks)
    for vec in vecs, c in vec
        for b in unique(unfixedblocks(c, blockindices))
            degree[b] += 1
        end
    end
    # Contiguous block ranges with roughly equal numbers of touching costs
    cumulative = cumsum(degree)
    total = isempty(cumulative) ? 0 : cumulative[end]
    ends = zeros(Int, N)
    for t in 1:N-1
        ends[t] = clamp(searchsortedfirst(cumulative, cld(total * t, N)), t == 1 ? 0 : ends[t - 1], nblocks)
    end
    ends[N] = nblocks
    ranges = [BlockRange((t == 1 ? 0 : ends[t - 1]) + 1, ends[t]) for t in 1:N]

    primary = [Int[] for _ in 1:nv, _ in 1:N]
    secondary = [Int[] for _ in 1:nv, _ in 1:N]
    for (i, vec) in enumerate(vecs)
        for (k, c) in enumerate(vec)
            blocks = unfixedblocks(c, blockindices)
            if isempty(blocks)
                push!(primary[i, 1], k)
                continue
            end
            owners = unique(searchsortedfirst(ends, b) for b in blocks)
            first_owner = searchsortedfirst(ends, first(blocks))
            for t in owners
                push!(t == first_owner ? primary[i, t] : secondary[i, t], k)
            end
        end
    end
    return BlockThreadPlan(collect(map(length, vecs)), ranges, primary, secondary, zeros(N))
end

# A plan if there is enough work to linearize on several threads, otherwise `nothing`
function blockthreadplan(problem::Model, blockindices, nblocks, numthreads::StaticInt)
    N = dynamic(numthreads)
    (N > 1 && countcosts(costwork, problem.costs) >= MIN_WORK_FOR_THREADED_LINEARIZATION) || return nothing
    return build_block_thread_plan(problem, blockindices, nblocks, N)
end

# A linear system that only writes the block columns of one thread
struct OwnedLS{L}
    ls::L
    own::BlockRange
end
getoffsets(cost, o::OwnedLS) = getoffsets(cost, o.ls)
function updatesymlinearsystem!(o::OwnedLS, g::AbstractVector, H::AbstractArray, vars, varflags, blockindices)
    updateb!(getb(o.ls), g, vars, varflags, o.ls.x.boffsets, blockindices, o.own)
    updatesymA!(getA(o.ls), H, vars, varflags, blockindices, o.own)
end

@inline block_chunk(ls, vars, ::Tuple{}, plan, i, t) = 0.0
@inline function block_chunk(ls, vars, vecs::Tuple, plan, i, t)
    vec = first(vecs)
    f = bindleadingargs(costgradhess!, ls, vars)
    c = sum(f, view(vec, plan.primary[i, t]); init=0.0)
    sum(f, view(vec, plan.secondary[i, t]); init=0.0) # The values belong to other threads
    return c + block_chunk(ls, vars, Base.tail(vecs), plan, i + 1, t)
end

# Arguments of the threads, in a struct because Polyester converts arrays captured by a `@batch` loop into pointer arrays
struct BlockThreadArgs{LS, PL, V, CV}
    ls::LS
    plan::PL
    vars::V
    vecs::CV
end
block_thread_task!(args::BlockThreadArgs, t::Integer)::Float64 = block_chunk(OwnedLS(args.ls, args.plan.ranges[t]), args.vars, args.vecs, args.plan, 1, t)

# Cost, gradient and Hessian into `ls` (a single, uncopied linear system) on several threads
function block_threaded!(ls, plan::Union{Nothing, BlockThreadPlan}, vars, costs::CostStruct)::Float64
    if plan === nothing || !plan_is_valid(plan, costs)
        zero!(ls)
        return costgradhess_nozero!(ls, vars, costs)
    end
    zero!(ls)
    args = BlockThreadArgs(ls, plan, vars, values(costs))
    @batch for t in 1:length(plan.ranges)
        plan.costs[t] = block_thread_task!(args, t)
    end
    return sum(plan.costs)
end
