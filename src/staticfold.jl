# The static single-variable system accumulated in values (improvements.md item 19, stage 1).
#
# For a static single-variable system (`UniVariateLSstatic`), an evaluation folds every cost's (cost, g, H) into
# immutable values, the gradient an `SVector` and the Hessian an `SHermitianCompact` (its lower triangle), and writes
# them once, both
# triangles, into the storage `fdf!` was handed. Nothing is zeroed and nothing mirrored there, and no entry of the
# system is loaded and stored per cost. `SingleVarCostGradHess` reaches it by dispatch on the system's type, so a
# dynamic system's path (`updatesymlinearsystem!`, `gradhesshelper!`) is unchanged.
#
# Every entry is the expression `addsys!` adds in place (`muladd` of the same operands, converted to the Hessian's
# element type per cost as a store into it converts), and the gradient is summed in `Float64`.
#
# The cost is summed in the order `sum` sums each cost vector (`foldvec` is Base's `mapreduce_impl`: sequential within
# blocks of `FOLD_BLOCK` costs, pairwise above), as the in-place path sums it, so `fdf!` returns the cost it returned
# before, bit for bit: the iterators compare it with the cost `f` returns.

# ---- the accumulator: the gradient, and the Hessian as its lower triangle (column-major). Measured on multilateration,
# d = 2 to 16, against a full SMatrix and a tuple of the triangle (item 19, M3): the compact form and the tuple were
# equal, and the full SMatrix 1.4 times slower at d = 6.
struct StaticSum{N, T, L}
    g::SVector{N, Float64}
    H::SHermitianCompact{N, T, L}
end
trianglesize(N) = (N * (N + 1)) >> 1
@inline zerosum(::Val{N}, ::Type{T}) where {N, T} =
    StaticSum{N, T, trianglesize(N)}(zero(SVector{N, Float64}), SHermitianCompact{N, T, trianglesize(N)}(zero(SVector{trianglesize(N), T})))
@inline zerosum(::LinearSystem{UniVariateLSstatic_x{N}, <:UniVariateLSstatic_ls{N, <:Any, T}}) where {N, T} = zerosum(Val(N), T)
@inline zerosum(ls::LinearSystemShared) = zerosum(LinearSystem(ls, 1))

# (i, j) of each entry of the lower triangle, column-major, as constants
@generated trianglepairs(::Val{N}) where N = Expr(:tuple, [:(($i, $j)) for j in 1:N for i in j:N]...)
# The entry of the triangle that holds (i, j) of the full matrix, as constants
@generated function trianglefull(::Val{N}) where N
    k(i, j) = i >= j ? (j - 1) * N - ((j - 1) * (j - 2)) >> 1 + (i - j + 1) : k(j, i)
    return Expr(:tuple, [k(i, j) for j in 1:N for i in 1:N]...)
end

# ---- one cost's (g, H) added to the values: the counterparts of addsys!
@inline addgrad(acc::SVector{N, Float64}, g) where N = SVector{N, Float64}(ntuple(i -> @inbounds(acc[i] + g[i]), Val(N)))

# One row's lazy product: dc u[i] u[j] + c2 gu[i] gu[j], as addsys! computes it
@inline function addhess(acc::SHermitianCompact{N, T, L}, H::GramLazy{<:Any, <:Adjoint}, ::Val{N}) where {L, T, N}
    a = acc.lowertriangle
    u = parent(H.J)
    P = trianglepairs(Val(N))
    return SHermitianCompact{N, T, L}(SVector{L, T}(ntuple(Val(L)) do k
        @inbounds begin
            i, j = P[k]
            T(muladd(u[i], H.dc * u[j], addc2(H, i, c2col(H, j), a[k])))
        end
    end))
end
# Several rows
@inline function addhess(acc::SHermitianCompact{N, T, L}, H::GramLazy{<:Any, <:StaticMatrix}, ::Val{N}) where {L, T, N}
    a = acc.lowertriangle
    J = H.J
    M = size(J, 1)
    P = trianglepairs(Val(N))
    return SHermitianCompact{N, T, L}(SVector{L, T}(ntuple(Val(L)) do k
        @inbounds begin
            i, j = P[k]
            s = addc2(H, i, c2col(H, j), a[k])
            for m in 1:M
                s = muladd(J[m, i], H.dc * J[m, j], s)
            end
            T(s)
        end
    end))
end
# Any other Hessian (a static matrix, the dynamic kernels' buffers, a user's computecostgradhess): read entry by entry
@inline function addhess(acc::SHermitianCompact{N, T, L}, H::AbstractMatrix, ::Val{N}) where {L, T, N}
    a = acc.lowertriangle
    P = trianglepairs(Val(N))
    return SHermitianCompact{N, T, L}(SVector{L, T}(ntuple(Val(L)) do k
        @inbounds begin
            i, j = P[k]
            T(a[k] + H[i, j])
        end
    end))
end
@inline addsum(acc::StaticSum{N, T, L}, g, H) where {N, T, L} = StaticSum{N, T, L}(addgrad(acc.g, g), addhess(acc.H, H, Val(N)))

# ---- one cost: the counterparts of costgradhess! and gradhesshelper!, returning (cost, sum)
@inline function foldcost(acc, ls, vars, cost)
    blockind = getoffsets(cost, ls)
    return foldhelper(acc, cost, getvars(cost, vars), computevarflags(blockind))
end
@inline function foldhelper(acc::S, cost, vars, varflags::StaticInt) where S
    # The result handed over whole, as in gradhesshelper!: where it is not inferred, one dispatch on the concrete tuple
    return foldresult(acc, computecostgradhess(varflags, cost, vars...))::Tuple{Float64, S}
end
@inline foldresult(acc, (c, g, H)) = (Float64(c), addsum(acc, g, H))
function foldhelper(acc, cost, vars, ::StaticInt{0})
    @warn "A cost for which all variables are fixed is being optimized. Remove such cost functions from the cost to avoid redundant computation, using the subproblem() method." maxlog=1
    return Float64(computecost(cost, vars...)), acc
end
@inline function foldhelper(acc::S, cost, vars, varflags::Integer) where S
    nv = ndeps(cost)
    maxflags = static(2 ^ dynamic(nv) - 1)
    varflags == maxflags && return foldhelper(acc, cost, vars, maxflags)
    if nv <= 5
        # Value dispatch on the varflags, the accumulator passed through and the same type returned
        return sirbsdispatch(static(0), maxflags - static(1), varflags, bindleadingargs(foldhelper, acc, cost, vars))::Tuple{Float64, S}
    end
    return foldhelper(acc, cost, vars, static(varflags))::Tuple{Float64, S}
end

# ---- the costs of one vector: Base's mapreduce_impl for `sum`, so the cost is summed as `sum` sums it
const FOLD_BLOCK = 1024 # Base.pairwise_blocksize
function foldvec(acc, ls, vars, v::AbstractVector, lo::Int, hi::Int)
    if lo == hi
        return @inbounds foldcost(acc, ls, vars, v[lo])
    elseif hi - lo < FOLD_BLOCK
        c, acc = @inbounds foldcost(acc, ls, vars, v[lo])
        c2, acc = @inbounds foldcost(acc, ls, vars, v[lo+1])
        c += c2
        for i in lo+2:hi
            ci, acc = @inbounds foldcost(acc, ls, vars, v[i])
            c += ci
        end
        return c, acc
    end
    mid = lo + (hi - lo) >> 1
    c1, acc = foldvec(acc, ls, vars, v, lo, mid)
    c2, acc = foldvec(acc, ls, vars, v, mid + 1, hi)
    return c1 + c2, acc
end
# Over a CostStruct (or a subset of each of its vectors), as `sum(f, costs; init=0.0)` and `sumsubset` do it
@inline foldtuple(c, acc, ls, vars, subset) = (c, acc)
@inline function foldtuple(c, acc, ls, vars, subset, v::AbstractVector, rest...)
    sub = subset(v)
    if isempty(sub)
        return foldtuple(c, acc, ls, vars, subset, rest...)
    end
    cv, acc = foldvec(acc, ls, vars, sub, firstindex(sub), lastindex(sub))
    return foldtuple(c + cv, acc, ls, vars, subset, rest...)
end
@inline foldcosts(acc, ls, vars, costs::CostStruct, subset=identity) = foldtuple(0.0, acc, ls, vars, subset, values(costs)...)

# ---- the single write: g, and both triangles of H, in the element types of what was handed over
@inline function writesum!(g, H, acc::StaticSum{N, T}) where {N, T}
    F = trianglefull(Val(N))
    lt = acc.H.lowertriangle
    copyto!(H, SMatrix{N, N, T}(ntuple(l -> @inbounds(lt[F[l]]), Val(N * N))))
    copyto!(g, acc.g)
    return nothing
end

# ---- the evaluation
function staticfold!(g, H, ls::LinearSystem, vars, costs::CostStruct)::Float64
    c, acc = foldcosts(zerosum(ls), ls, vars, costs)
    writesum!(g, H, acc)
    return c
end

# On several threads: each task folds its chunk of every cost vector and writes it once into its own copy of the
# system; the copies are summed as on the in-place path. Each copy holds both triangles, so the sum needs no mirror.
function staticfold!(g, H, ls::LinearSystemShared{F, V, N}, vars, costs::CostStruct)::Float64 where {F, V, N}
    if H isa typeof(ls.ls[1].A) && g isa typeof(ls.ls[1].b)
        # The optimizer's storage is the first copy: the sum lands in it
        target = replace_first_ls(ls, makels(ls.ls[1], H, g))
        return staticfold_shared!(target, vars, costs)
    end
    c = staticfold_shared!(ls, vars, costs)
    copyto!(g, ls.ls[1].b)
    copyto!(H, ls.ls[1].A)
    return c
end
function staticfold_shared!(linsystem::LinearSystemShared{F, V, N}, vars, costs::CostStruct)::Float64 where {F, V, N}
    args = threadargs((linsystem, vars, costs))
    GC.@preserve linsystem vars costs begin
        @batch for taskid in 1:N
            linsystem_, vars_, costs_ = fromthreadargs(args)
            ls = LinearSystem(linsystem_, taskid)
            c, acc = foldcosts(zerosum(ls), ls, vars_, costs_, bindleadingargs(ithchunk, N, taskid))
            writesum!(ls.ls.b, ls.ls.A, acc)
            linsystem_.costs[taskid] = c
        end
        # The tree reduction of the in-place path (costgradhess! of a LinearSystemShared), unchanged
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
