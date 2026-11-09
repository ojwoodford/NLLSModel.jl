using StaticArrays, Static, SparseArrays

SR(first, last) = StaticArrays.SUnitRange(dynamic(first), dynamic(last))

struct Bind{F, Args<:Tuple}
    f::F
    args::Args
end
@inline (b::Bind)(xs...) = b.f(b.args..., xs...)
@inline bindleadingargs(f, args...) = Bind(f, args)

macro bitiset(flags, bit)
    esc(:(((1 << ($bit - 1)) & $flags) != 0))
end

bitiset(flags::StaticInt, bit) = (static(1 << (bit - 1)) & flags) != static(0)
bitiset(flags, bit) = (1 << (bit - 1)) & flags != 0

function runlengthencodesortedints(sortedints)
    runindices = Vector{Int}(undef, sortedints[end]+2)
    ind = 1
    currval = 1
    runindices[currval] = ind
    for val in sortedints
        while val >= currval
            currval += 1
            runindices[currval] = ind
        end
        ind += 1
    end
    runindices[currval+1] = ind
    return runindices
end

function block_sparsity(problem, unfixed_inds::AbstractVector{<:Integer})
    # Compute the block sparsity for the given unfixed variable ordering
    n = length(unfixed_inds)
    sparsity = getvarcostmap(problem)[unfixed_inds, :]
    sparsity = triu(sparse(sparsity * sparsity' .> 0) + sparse(1:n, 1:n, true, n, n) .> 0)
    return sparsity
end
block_sparsity(problem, unfixed::AbstractVector{Bool}) = block_sparsity(problem, findall(unfixed))
block_sparsity(problem, unfixed) = block_sparsity(problem, convert(Vector{Int}, findall(unfixed)))

function block_sizes_indices(variables, unfixed, nblocks)
    unfixed_raw = findall(unfixed)
    dims = [UInt8(dynamic(nvars(variables[i]))) for i in unfixed_raw]

    # Check if dimensions are already grouped into contiguous runs
    is_grouped = true
    seen = Set{UInt8}()
    prev = UInt8(0)
    for d in dims
        if d != prev
            if d in seen
                is_grouped = false
                break
            end
            push!(seen, d)
            prev = d
        end
    end

    if is_grouped
        unfixed_inds = unfixed_raw
        rle = BlockRLE(dims)
        blockindices = zeros(UInt, length(variables))
        for (b, var_idx) in enumerate(unfixed_raw)
            blockindices[var_idx] = b
        end
        return rle, blockindices, unfixed_inds, true
    else
        # Stable sort by dimension (descending: 6, then 3, etc.) to group identical dimensions into RLE runs
        perm = sortperm(dims; rev=true, alg=Base.Sort.MergeSort)
        unfixed_inds = unfixed_raw[perm]
        sorted_dims = dims[perm]
        rle = BlockRLE(sorted_dims)
        blockindices = zeros(UInt, length(variables))
        for (b, var_idx) in enumerate(unfixed_inds)
            blockindices[var_idx] = b
        end
        return rle, blockindices, unfixed_inds, false
    end
end

"""
    sparse_dense_decision(ndims, sparsity, blocksizes) -> Bool
    sparse_dense_decision(ndims, nnz) -> Bool

Heuristic returning `true` when a symmetric linear system of dimension `ndims` with lower-triangular non-zero count `nnz`
(computed from a block `sparsity` pattern and `blocksizes` via `BlockLinearSystems.block_sparse_nnz`) is faster to factorize
as a sparse matrix than as a dense matrix: `64 * nnz < 25 * ndims * (ndims - 40)`.

Used by `optimvars_dispatch` to choose between a `SparseBlockLinearSystem` (sparse LDLᵀ) and a `DenseBlockLinearSystem` (dense Cholesky) linear system.
"""
sparse_dense_decision(ndims, sparsity, blocksizes) = sparse_dense_decision(ndims, block_sparse_nnz(sparsity, blocksizes))
sparse_dense_decision(ndims, nnz) = (nnz * 64) < (25 * ndims * (ndims - 40)) # Threshold nnz (for lower triangle) = 25/64 * (d^2 - 40d)
