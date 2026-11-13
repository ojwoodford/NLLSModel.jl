using Polyester

# Threaded linearization of Schur problems without whole-system copies.
#
# Each eliminated variable owns its block of E and its coupling blocks of C, and every cost touches at most
# one of them. If each thread is given whole eliminated variables (all costs that involve them), threads write disjoint parts of
# `E_data`, `C_data` and the eliminated part of the gradient, so those are shared. Only `R` (retained-retained blocks)
# and the retained part of the gradient can be written by several threads, so only those are per-thread copies, and
# only they need summing afterwards.
#
# To make an eliminated variable's costs contiguous, `build_schur_thread_plan!` sorts every cost vector of the problem by eliminated variable
# (a stable sort; costs that involve no unfixed eliminated variable go last). This reorders `problem.costs` in place.

struct SchurThreadPlan
    veclengths::Vector{Int}          # Cost vector lengths the plan was built for (to detect later changes)
    owned::Matrix{UnitRange{Int}}    # [cost vector, thread] -> costs of the eliminated variables owned by the thread
    tail::Matrix{UnitRange{Int}}     # [cost vector, thread] -> share of the costs without an eliminated variable
end

plan_is_valid(plan::SchurThreadPlan, costs::CostStruct) = plan.veclengths == collect(map(length, values(costs)))

# Index (1:neblocks) of the unfixed eliminated variable of a cost, or 0 if it has none
function eliminatedkey(cost, blockindices, nrblocks)
    vinds = varindices(cost)
    if vinds isa Integer
        b = Int(blockindices[vinds])
        return b > nrblocks ? b - nrblocks : 0
    end
    for v in vinds
        b = Int(blockindices[v])
        b > nrblocks && return b - nrblocks
    end
    return 0
end

function build_schur_thread_plan!(problem::Model, blockindices, nrblocks::Integer, neblocks::Integer, N::Integer)
    vecs = values(problem.costs)
    nv = length(vecs)
    sortedkeys = Vector{Vector{Int}}(undef, nv)
    counts = zeros(Int, neblocks)
    for (i, vec) in enumerate(vecs)
        keys = [eliminatedkey(c, blockindices, nrblocks) for c in vec]
        sortkeys = [k == 0 ? typemax(Int) : k for k in keys]
        perm = sortperm(sortkeys; alg=Base.Sort.MergeSort) # Stable
        permute!(vec, perm)
        sortedkeys[i] = sortkeys[perm]
        for k in sortedkeys[i]
            k != typemax(Int) && (counts[k] += 1)
        end
    end
    problem.varcostmapvalid = false # Cost order has changed

    # Split the eliminated variables into N contiguous ranges holding roughly equal numbers of costs
    cumulative = cumsum(counts)
    total = isempty(cumulative) ? 0 : cumulative[end]
    bound = zeros(Int, N + 1) # Landmarks owned by thread t are bound[t]+1 : bound[t+1]
    for t in 1:N-1
        bound[t + 1] = clamp(searchsortedfirst(cumulative, cld(total * t, N)), bound[t], neblocks)
    end
    bound[N + 1] = neblocks

    owned = Matrix{UnitRange{Int}}(undef, nv, N)
    tail = Matrix{UnitRange{Int}}(undef, nv, N)
    for i in 1:nv
        sk = sortedkeys[i]
        tailstart = searchsortedfirst(sk, typemax(Int))
        ntail = length(sk) - tailstart + 1
        for t in 1:N
            owned[i, t] = (searchsortedlast(sk, bound[t]) + 1):searchsortedlast(sk, bound[t + 1])
            tail[i, t] = (tailstart - 1) .+ chunk_indices(ntail, N, t)
        end
    end
    return SchurThreadPlan(collect(map(length, vecs)), owned, tail)
end

# Per-thread buffer for the threaded Schur linearization: own R and gradient; E and C are shared at evaluation time.
function schur_thread_buffer(ls::MultiVariateLSSchur_ls)
    A = ls.A
    R_copy = A.R isa Matrix ? similar(A.R) : copy_structure(A.R)
    return MultiVariateLSSchur_ls(SchurBlockMatrix(A.structure, R_copy, Float64[], Float64[]), zeros(Float64, length(ls.b)))
end

# Like `sharedlinearsystem`, but the extra buffers carry no E and C storage
function sharedschurlinearsystem(ls::LinearSystem{F, V}, problem, numthreads::StaticInt{N}) where {F, V, N}
    (N == 1 || countcosts(costwork, problem.costs) < MIN_WORK_FOR_THREADED_LINEARIZATION) && return ls
    buffers = SVector{N, V}(ntuple(i -> i == 1 ? ls.ls : schur_thread_buffer(ls.ls), Val(N)))
    return LinearSystemShared{F, V, N}(ls.x, buffers, MVector{N, Float64}(undef))
end

zero_schur_R!(R::Matrix) = fill!(R, 0.0)
zero_schur_R!(R) = zero!(R)
add_schur_R!(dest::Matrix, src::Matrix) = (dest .+= src; nothing)
add_schur_R!(dest, src) = (dest.data .+= src.data; nothing)

@inline schur_chunk(ls, vars, ::Tuple{}, plan, i, t) = 0.0
@inline function schur_chunk(ls, vars, vecs::Tuple, plan, i, t)
    vec = first(vecs)
    f = bindleadingargs(costgradhess!, ls, vars)
    c = sum(f, view(vec, plan.owned[i, t]); init=0.0) + sum(f, view(vec, plan.tail[i, t]); init=0.0)
    return c + schur_chunk(ls, vars, Base.tail(vecs), plan, i + 1, t)
end

# Work of one thread: zero its own R and gradient, then linearize its costs
# Arguments of the threads, in a struct because Polyester converts arrays captured by a `@batch` loop into pointer arrays
struct SchurThreadArgs{LS, PL, HT, G, V, CV}
    linsystem::LS
    plan::PL
    H::HT
    g::G
    vars::V
    vecs::CV
end

function schur_thread_task!(args::SchurThreadArgs{LinearSystemShared{F, V, N}}, t::Integer)::Float64 where {F, V, N}
    linsystem, plan, H, g, vars, vecs = args.linsystem, args.plan, args.H, args.g, args.vars, args.vecs
    ls_t = if t == 1
        MultiVariateLSSchur_ls(H, g)
    else
        buf = linsystem.ls[t]
        MultiVariateLSSchur_ls(SchurBlockMatrix(H.structure, buf.A.R, H.E_data, H.C_data), buf.b)
    end
    zero_schur_R!(ls_t.A.R)
    fill!(ls_t.b, 0.0)
    return schur_chunk(LinearSystem{F, V}(linsystem.x, ls_t), vars, vecs, plan, 1, t)
end

# Cost, gradient and Hessian of a Schur problem on N threads
function schur_threaded!(linsystem::LinearSystemShared{F, V, N}, plan, H::SchurBlockMatrix, g, vars, costs::CostStruct)::Float64 where {F, V, N}
    if plan === nothing || !plan_is_valid(plan, costs)
        # Costs changed since the plan was made: fall back to one thread
        ls = LinearSystem{F, V}(linsystem.x, MultiVariateLSSchur_ls(H, g))
        zero!(ls)
        return costgradhess_nozero!(ls, vars, costs)
    end

    # E and C are shared between threads, so zero them once before the threads start writing
    fill!(H.E_data, 0.0)
    fill!(H.C_data, 0.0)
    vecs = values(costs)
    args = SchurThreadArgs(linsystem, plan, H, g, vars, vecs)
    @batch for t in 1:N
        linsystem.costs[t] = schur_thread_task!(args, t)
    end

    # Only R and the gradient were written by more than one thread
    for t in 2:N
        buf = linsystem.ls[t]
        add_schur_R!(H.R, buf.A.R)
        g .+= buf.b
    end
    total = 0.0
    for t in 1:N
        total += linsystem.costs[t]
    end
    return total
end
