# Threaded linearization of Schur problems without whole-system copies.
#
# Each eliminated variable owns its block of E and its coupling blocks of C, and every cost touches at most
# one of them: BlockLinearSystems' `addblock!` documents that calls for pairs involving different eliminated blocks write
# disjoint storage. If each thread is given whole eliminated variables (all costs that involve them), threads write disjoint parts of
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

# Per-thread buffer for the threaded Schur linearization: own R and gradient; E and C are those of the system it is made
# from, and at evaluation time those of the system being filled.
function schur_thread_buffer(ls::MultiVariateLSSchur_ls)
    A = ls.A
    R_copy = A.R isa Matrix ? similar(A.R) : copy_structure(A.R)
    return MultiVariateLSSchur_ls(SchurBlockHessian(R_copy, A), zeros(Float64, length(ls.b)))
end

# Like `sharedlinearsystem`, but the extra buffers have no E and C storage of their own
# One buffer per thread of `normalthreads`
function sharedschurlinearsystem(ls::LinearSystem, work::Integer, numthreads::StaticInt, costsplit::Integer, min_work=MIN_WORK_PER_THREAD_NORMAL)
    n = normalthreads(work, min_work, dynamic(numthreads), costsplit)
    n == 1 && return ls
    return sharedschurlinearsystem(ls, static(n))
end
function sharedschurlinearsystem(ls::LinearSystem{F, V}, ::StaticInt{N}) where {F, V, N}
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
# Arguments of the threads, passed to the `@batch` loop through `threadargs`
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
        MultiVariateLSSchur_ls(SchurBlockHessian(buf.A.R, H), buf.b)
    end
    zero_schur_R!(ls_t.A.R)
    fill!(ls_t.b, 0.0)
    return schur_chunk(LinearSystem{F, V}(linsystem.x, ls_t), vars, vecs, plan, 1, t)
end

# Cost, gradient and Hessian of a Schur problem on N threads
function schur_threaded!(linsystem::LinearSystemShared{F, V, N}, plan, H::SchurBlockHessian, g, vars, costs::CostStruct)::Float64 where {F, V, N}
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
    targs = threadargs(args)
    GC.@preserve args begin
        @batch for t in 1:N
            args_ = fromthreadargs(targs)
            args_.linsystem.costs[t] = schur_thread_task!(args_, t)
        end
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
