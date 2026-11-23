struct LinearSystem{F, V}
    x::F
    ls::V
end
LinearSystem{F, V}(xargs::Tuple, lsargs::Tuple) where {F, V} = LinearSystem{F, V}(F(xargs...), V(lsargs...))
updateunfixed(old::LinearSystem{F, V}, unfixed) where {F, V} = LinearSystem{F, V}(F(unfixed, old.x), old.ls)

struct LinearSystemShared{F, V, N}
    x::F
    ls::SVector{N, V}
    costs::MVector{N, Float64}
end
function LinearSystemShared{F, V, N}(xargs::Tuple, lsargs::Tuple) where {F, V, N}
    return LinearSystemShared{F, V, N}(F(xargs...), SVector{N, V}(ntuple(i -> V(lsargs...), Val(N))), MVector{N, Float64}(undef))
end
updateunfixed(old::LinearSystemShared{F, V, N}, unfixed) where {F, V, N} = LinearSystemShared{F, V, N}(F(unfixed, old.x), old.ls, old.costs)

# Ephemeral task wrapper constructed on the stack
LinearSystem(ls::LinearSystemShared{F, V, N}, ind) where {F, V, N} = LinearSystem{F, V}(ls.x, ls.ls[ind])

@inline function replace_first_ls(ls_shared::LinearSystemShared{F, V, N}, ls1::V) where {F, V, N}
    buffers = Base.setindex(ls_shared.ls, ls1, 1)
    return LinearSystemShared{F, V, N}(ls_shared.x, buffers, ls_shared.costs)
end

getA(ls::LinearSystem) = ls.ls.A
getb(ls::LinearSystem) = ls.ls.b
getx(ls::LinearSystem) = ls.x.x

getA(ls::LinearSystemShared) = ls.ls[1].A
getb(ls::LinearSystemShared) = ls.ls[1].b
getx(ls::LinearSystemShared) = ls.x.x

getA(ls) = ls.A
getb(ls) = ls.b
getx(ls) = ls.x

# Uni-variate linear system. `T` is the element type of the Hessian (the `precision` keyword, single-precision.md); the
# gradient is Float64 whatever it is.
struct UniVariateLSdynamic_ls{T}
    A::Matrix{T}
    b::Vector{Float64}

    function UniVariateLSdynamic_ls{T}(varlen) where T
        return new{T}(Matrix{T}(undef, varlen, varlen), Vector{Float64}(undef, varlen))
    end
    UniVariateLSdynamic_ls(A::Matrix{T}, b::Vector{Float64}) where T = new{T}(A, b)
end
UniVariateLSdynamic_ls(varlen::Integer) = UniVariateLSdynamic_ls{Float64}(varlen)

struct UniVariateLSdynamic_x
    varindex::UInt
    x::Vector{Float64}

    function UniVariateLSdynamic_x(unfixed, varlen)
        return new(UInt(unfixed), Vector{Float64}(undef, varlen))
    end
    function UniVariateLSdynamic_x(unfixed, old::UniVariateLSdynamic_x)
        return new(UInt(unfixed), old.x)
    end
end

struct UniVariateLSstatic_ls{N, N2, T}
    A::MMatrix{N, N, T, N2}
    b::MVector{N, Float64}

    function UniVariateLSstatic_ls{N, N2, T}() where {N, N2, T}
        return new{N, N2, T}(MMatrix{N, N, T, N2}(undef), MVector{N, Float64}(undef))
    end
    UniVariateLSstatic_ls(A::MMatrix{N, N, T, N2}, b::MVector{N, Float64}) where {N, N2, T} = new{N, N2, T}(A, b)
end
UniVariateLSstatic_ls{N, N2}() where {N, N2} = UniVariateLSstatic_ls{N, N2, Float64}()

struct UniVariateLSstatic_x{N}
    varindex::UInt
    x::MVector{N, Float64}

    function UniVariateLSstatic_x{N}(unfixed) where N
        return new(UInt(unfixed), MVector{N, Float64}(undef))
    end
    function UniVariateLSstatic_x{N}(unfixed, old::UniVariateLSstatic_x{N}) where N
        return new(UInt(unfixed), old.x)
    end
end

const UniVariateLSsingle = Union{LinearSystem{UniVariateLSdynamic_x, <:UniVariateLSdynamic_ls}, LinearSystem{UniVariateLSstatic_x{N}, <:UniVariateLSstatic_ls{N}} where N}
const UniVariateLSdynamic = Union{LinearSystem{UniVariateLSdynamic_x, <:UniVariateLSdynamic_ls}, LinearSystemShared{UniVariateLSdynamic_x, V, M} where {V <: UniVariateLSdynamic_ls, M}}
const UniVariateLSstatic = Union{LinearSystem{UniVariateLSstatic_x{N}, <:UniVariateLSstatic_ls{N}} where N, LinearSystemShared{UniVariateLSstatic_x{N}, V, M} where {N, V <: UniVariateLSstatic_ls{N}, M}}
const UniVariateLS = Union{UniVariateLSdynamic, UniVariateLSstatic}

# Multi-variate linear system
struct MultiVariateLSsparse_x{T}
    x::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{Int} # One per block in b
    hessian::SparseMatrixCSC{T, Int} # Storage for sparse hessian
    sparseindices::Vector{Int}
end

struct MultiVariateLSsparse_ls{T}
    A::SparseBlockLinearSystem{T}
    b::Vector{Float64}
end
# The gradient converted to a Vector{Float64}, as the non-parametric type did
MultiVariateLSsparse_ls(A::SparseBlockLinearSystem{T}, b::AbstractVector) where T = MultiVariateLSsparse_ls{T}(A, b)

struct MultiVariateLSdense_x
    x::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{Int} # One per block in b
end

struct MultiVariateLSdense_ls{T, M<:AbstractMatrix{T}}
    A::DenseBlockLinearSystem{T, M}
    b::Vector{Float64}
end
MultiVariateLSdense_ls(A::DenseBlockLinearSystem{T, M}, b::AbstractVector) where {T, M} = MultiVariateLSdense_ls{T, M}(A, b)

# Schur linear system
struct MultiVariateLSSchur_x
    x::Vector{Float64}
    blockindices::Vector{UInt}
    boffsets::Vector{Int}
    structure::SchurStructure
    unfixed_inds::Vector{Int}
end

struct MultiVariateLSSchur_ls{T, BType}
    A::SchurBlockHessian{T, BType}
    b::Vector{Float64}
end
MultiVariateLSSchur_ls(A::SchurBlockHessian{T, BType}, b::AbstractVector) where {T, BType} = MultiVariateLSSchur_ls{T, BType}(A, b)

const MultiVariateLSsparse = Union{LinearSystem{<:MultiVariateLSsparse_x, <:MultiVariateLSsparse_ls}, LinearSystemShared{X, V, M} where {X <: MultiVariateLSsparse_x, V <: MultiVariateLSsparse_ls, M}}
const MultiVariateLSdense = Union{LinearSystem{MultiVariateLSdense_x, <:MultiVariateLSdense_ls}, LinearSystemShared{MultiVariateLSdense_x, <:MultiVariateLSdense_ls, M} where M}
const MultiVariateLSSchur{BType} = Union{LinearSystem{MultiVariateLSSchur_x, <:MultiVariateLSSchur_ls{<:Any, BType}}, LinearSystemShared{MultiVariateLSSchur_x, V, M} where {V <: MultiVariateLSSchur_ls{<:Any, BType}, M}}
const MultiVariateLSsingle = Union{LinearSystem{<:MultiVariateLSsparse_x, <:MultiVariateLSsparse_ls}, LinearSystem{MultiVariateLSdense_x, <:MultiVariateLSdense_ls}, LinearSystem{MultiVariateLSSchur_x, <:MultiVariateLSSchur_ls}}
const MultiVariateLS = Union{MultiVariateLSsparse, MultiVariateLSdense, MultiVariateLSSchur}

function MultiVariateLSsparse(A::SparseBlockLinearSystem{T}, blockindices, storehessian=true) where T
    @assert A.rowblocksizes == A.columnblocksizes
    boffsets = computestartindices(A.rowblocksizes)
    blen = boffsets[end] + A.rowblocksizes[end] - 1
    if storehessian
        x = Vector{Float64}(undef, blen)
        hessian, sparseindices = sparse_with_map(A; symmetric=true)
    else
        x = Vector{Float64}(undef, 1)
        sparseindices = Vector{Int}()
        hessian = spzeros(T, 1, 1)
    end
    return LinearSystem(
        MultiVariateLSsparse_x(x, blockindices, boffsets, hessian, sparseindices),
        MultiVariateLSsparse_ls(A, zeros(Float64, blen))
    )
end

function MultiVariateLSsparse(A::SparseBlockLinearSystem, b, x, blockindices, boffsets, hessian, sparseindices)
    return LinearSystem(
        MultiVariateLSsparse_x(x, blockindices, boffsets, hessian, sparseindices),
        MultiVariateLSsparse_ls(A, b)
    )
end

function MultiVariateLSdense(blocksizes, blockindices, ::Type{T}=Float64) where T
    boffsets = computestartindices(blocksizes)
    blen = boffsets[end] + blocksizes[end]
    push!(boffsets, blen)
    blen -= 1
    return LinearSystem(
        MultiVariateLSdense_x(Vector{Float64}(undef, blen), blockindices, boffsets),
        MultiVariateLSdense_ls(DenseBlockLinearSystem{T}(blocksizes), zeros(Float64, blen))
    )
end

function MultiVariateLSdense(A::DenseBlockLinearSystem, b, x, blockindices, boffsets)
    return LinearSystem(
        MultiVariateLSdense_x(x, blockindices, boffsets),
        MultiVariateLSdense_ls(A, b)
    )
end

BlockLinearSystems.copy_structure(ls::UniVariateLSdynamic_ls{T}) where T = UniVariateLSdynamic_ls{T}(size(ls.A, 1))
BlockLinearSystems.copy_structure(::UniVariateLSstatic_ls{N, N2, T}) where {N, N2, T} = UniVariateLSstatic_ls{N, N2, T}()
BlockLinearSystems.copy_structure(ls::MultiVariateLSdense_ls) = MultiVariateLSdense_ls(copy_structure(ls.A), zeros(Float64, length(ls.b)))
BlockLinearSystems.copy_structure(ls::MultiVariateLSsparse_ls) = MultiVariateLSsparse_ls(copy_structure(ls.A), zeros(Float64, length(ls.b)))
BlockLinearSystems.copy_structure(ls::MultiVariateLSSchur_ls) = MultiVariateLSSchur_ls(copy_structure(ls.A), zeros(Float64, length(ls.b)))
BlockLinearSystems.copy_structure(ls::LinearSystem{F, V}) where {F, V} = LinearSystem{F, V}(ls.x, copy_structure(ls.ls))
BlockLinearSystems.copy_structure(ls::LinearSystemShared{F, V, N}) where {F, V, N} = LinearSystemShared{F, V, N}(ls.x, SVector{N, V}(ntuple(i -> copy_structure(ls.ls[i]), Val(N))), MVector{N, Float64}(undef))

sharedlinearsystem(ls::LinearSystem, ::StaticInt{1}) = ls
sharedlinearsystem(ls::LinearSystemShared, ::StaticInt) = ls
function sharedlinearsystem(ls::LinearSystem{F, V}, ::StaticInt{N}) where {F, V, N}
    buffers = SVector{N, V}(ntuple(i -> i == 1 ? ls.ls : copy_structure(ls.ls), Val(N)))
    costs = MVector{N, Float64}(undef)
    return LinearSystemShared{F, V, N}(ls.x, buffers, costs)
end
sharedlinearsystem(ls, n::Integer) = sharedlinearsystem(ls, static(n))

# Each extra thread needs a full copy of the linear system (a dense Hessian is O(n^2)). Cap the memory the extra copies
# may use, reducing the number of threads that linearize rather than allocating without bound.
const MAX_THREAD_COPY_BYTES = 2^30
threadcopylimit(ls, nthreads::Integer, budget::Integer=MAX_THREAD_COPY_BYTES) = clamp(1 + budget ÷ max(1, Base.summarysize(ls.ls)), 1, nthreads)
# `work` is the Hessian-fill estimate of `workestimates`: one copy per thread of `hessianthreads`, capped by the memory budget
function sharedlinearsystem(ls, work::Integer, numthreads::StaticInt, costsplit::Integer, min_work=MIN_WORK_PER_THREAD_HESSIAN)
    n = hessianthreads(work, min_work, dynamic(numthreads), costsplit)
    n == 1 && return ls
    return sharedlinearsystem(ls, static(threadcopylimit(ls, n)))
end

add!(dest::UniVariateLSdynamic_ls, src::UniVariateLSdynamic_ls) = (dest.A .+= src.A; dest.b .+= src.b; nothing)
add!(dest::UniVariateLSstatic_ls, src::UniVariateLSstatic_ls) = (dest.A .+= src.A; dest.b .+= src.b; nothing)
add!(dest::MultiVariateLSdense_ls, src::MultiVariateLSdense_ls) = (dest.A.data .+= src.A.data; dest.b .+= src.b; nothing)
add!(dest::MultiVariateLSsparse_ls, src::MultiVariateLSsparse_ls) = (dest.A.data .+= src.A.data; dest.b .+= src.b; nothing)
function add!(dest::MultiVariateLSSchur_ls, src::MultiVariateLSSchur_ls)
    dest.b .+= src.b
    if dest.A.R isa Matrix
        dest.A.R .+= src.A.R
    else
        dest.A.R.data .+= src.A.R.data
    end
    dest.A.E_data .+= src.A.E_data
    dest.A.C_data .+= src.A.C_data
    return nothing
end
add!(dest::LinearSystem, src::LinearSystem) = add!(dest.ls, src.ls)

# The end of an evaluation: a dynamic single-variable system accumulated in its lower triangle (from the lazy product)
# is made symmetric here, once. A full H (a user's computecostgradhess, the dynamic kernels) has added both triangles;
# mirroring the lower one over the upper keeps the same values when H is symmetric. A static single-variable system is
# never accumulated in place: its costs are folded in values and written once, both triangles (src/staticfold.jl).
finish!(linsystem) = linsystem
function finish!(linsystem::UniVariateLSdynamic)
    A = getA(linsystem)
    n = size(A, 1)
    @inbounds for j in 2:n, i in 1:j-1
        A[i, j] = A[j, i]
    end
    return linsystem
end

function updatesymlinearsystem!(linsystem::UniVariateLSdynamic, g, H, unusedargs...)
    # Update the blocks in the problem
    addsys!(getb(linsystem), getA(linsystem), g, H)
end
# A full H (the dynamic kernels, a user's computecostgradhess): explicit loops, since the broadcast into an MMatrix
# system is not vectorized
function addsys!(b, A, g, H)
    n = length(b)
    @inbounds @simd for i in 1:n
        b[i] += g[i]
    end
    @inbounds for j in 1:n
        @simd for i in 1:n
            A[i, j] += H[i, j]
        end
    end
    return nothing
end

function updatesymA!(A, a, vars, varflags, blockindices)
    # Update the blocks in the problem
    loffseti = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            nvi = nvars(vars[i])
            rangei = blockrange(loffseti + static(1), nvi)
            loffseti += nvi
            @inbounds block(A, blockindices[i], blockindices[i], nvi, nvi) .+= view(a, rangei, rangei)

            loffsetj = static(0)
            @unroll for j in 1:i-1
                if bitiset(varflags, j)
                    @inbounds nvj = nvars(vars[j])
                    rangej = blockrange(loffsetj + static(1), nvj)
                    loffsetj += nvj
                    if @inbounds blockindices[i] >= blockindices[j] # Make sure the block matrix is lower triangular
                        @inbounds block(A, blockindices[i], blockindices[j], nvi, nvj) .+= view(a, rangei, rangej)
                    else
                        @inbounds block(A, blockindices[j], blockindices[i], nvj, nvi) .+= view(a, rangej, rangei)
                    end
                end
            end
        end
    end
end

function updateb!(B, b, vars, varflags, boffsets, blockindices)
    # Update the blocks in the problem
    loffset = static(1)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            nv = nvars(vars[i])
            @inbounds view(B, blockrange(boffsets[blockindices[i]], nv)) .+= view(b, blockrange(loffset, nv))
            loffset += nv
        end
    end
end

function updatesymlinearsystem!(linsystem::MultiVariateLS, g::AbstractVector, H::AbstractArray, vars, varflags, blockindices)
    updateb!(getb(linsystem), g, vars, varflags, linsystem.x.boffsets, blockindices)
    updatesymA!(getA(linsystem), H, vars, varflags, blockindices)
end

# One cost's gradient and Hessian into a Schur system: each block of the cost's Hessian is added to the system by naming
# the pair of variables, once for each variable with itself and once for each pair `j < i`. BlockLinearSystems decides
# where and in which orientation the pair is stored. No cost has two unfixed eliminated variables (the partition's
# check), so every pair named has storage.
function updatesymschur!(H_schur::SchurBlockHessian, g_full::Vector{Float64}, g::AbstractVector, H::AbstractArray, vars, varflags, boffsets, blockindices)
    updateb!(g_full, g, vars, varflags, boffsets, blockindices)

    loffseti = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            bi = Int(blockindices[i])
            nvi = nvars(vars[i])
            rangei = blockrange(loffseti + static(1), nvi)
            loffseti += nvi
            @inbounds addblock!(H_schur, bi, bi, H, rangei, rangei)

            loffsetj = static(0)
            @unroll for j in 1:i-1
                if bitiset(varflags, j)
                    bj = Int(blockindices[j])
                    nvj = nvars(vars[j])
                    rangej = blockrange(loffsetj + static(1), nvj)
                    loffsetj += nvj
                    @inbounds addblock!(H_schur, bi, bj, H, rangei, rangej)
                end
            end
        end
    end
end

function updatesymlinearsystem!(linsystem::MultiVariateLSSchur, g::AbstractVector, H::AbstractArray, vars, varflags, blockindices)
    updatesymschur!(getA(linsystem), getb(linsystem), g, H, vars, varflags, linsystem.x.boffsets, blockindices)
end

getgrad(ls::LinearSystem) = ls.ls.b
getgrad(ls::LinearSystemShared) = ls.ls[1].b
getgrad(linsystem) = getb(linsystem)

gethessian(ls::LinearSystem) = gethessian(ls.x, ls.ls)
gethessian(ls::LinearSystemShared) = gethessian(ls.x, ls.ls[1])
gethessian(::UniVariateLSdynamic_x, ls::UniVariateLSdynamic_ls) = ls.A
gethessian(::UniVariateLSstatic_x, ls::UniVariateLSstatic_ls) = ls.A
gethessian(::MultiVariateLSdense_x, ls::MultiVariateLSdense_ls) = symmetrifyfull(ls.A)
function gethessian(x::MultiVariateLSsparse_x, ls::MultiVariateLSsparse_ls)
    copyto!(x.hessian, ls.A, x.sparseindices)
    return x.hessian
end
gethessian(::MultiVariateLSSchur_x, ls::MultiVariateLSSchur_ls) = ls.A
gethessian(linsystem) = getA(linsystem)
gethessgrad(linsystem) = gethessian(linsystem), getgrad(linsystem)

function BlockLinearSystems.zero!(linsystem::Union{LinearSystem, LinearSystemShared})
    fill!(getb(linsystem), 0)
    zero!(getA(linsystem))
end

getoffsets(block, linsystem::MultiVariateLS) = @inbounds(linsystem.x.blockindices[varindices(block)])
function getoffsets(block, linsystem::UniVariateLS)
    varind = varindices(block)
    if isa(varind, Number)
        return SVector(UInt(varind == linsystem.x.varindex))
    end
    return convert.(UInt, varind .== linsystem.x.varindex)
end

function OptimizationVariableInterface.update!(to::Vector, from::Vector, linsystem::MultiVariateLS, step=getx(linsystem))
    # Update each variable
    for (i, j) in enumerate(linsystem.x.blockindices)
        if j != 0
            @inbounds to[i] = update(from[i], step, linsystem.x.boffsets[j])
        end
    end
end

@inline function OptimizationVariableInterface.update!(to::Vector, from::Vector, linsystem::UniVariateLS, step=getx(linsystem))
    # Update one variable
    @inbounds to[linsystem.x.varindex] = update(from[linsystem.x.varindex], step)
end
