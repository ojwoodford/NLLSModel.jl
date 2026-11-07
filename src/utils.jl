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

function block_sparsity(problem, unfixed)
    # Compute the block sparsity
    n = sum(unfixed)
    sparsity = getvarcostmap(problem)
    sparsity = sparsity[unfixed,:]
    sparsity = triu(sparse(sparsity * sparsity' .> 0) + sparse(1:n, 1:n, true, n, n) .> 0)
    return sparsity
end

function block_sizes_indices(variables, unfixed, nblocks)
    blockindices = zeros(UInt, length(variables))
    blocksizes = zeros(UInt, nblocks)
    nblocks = 0
    for (index, unfixed_) in enumerate(unfixed)
        if unfixed_
            nblocks += 1
            blockindices[index] = nblocks
            blocksizes[nblocks] = nvars(variables[index])
        end
    end
    return blocksizes, blockindices
end

"""
    sparse_dense_decision(ndims, sparsity, blocksizes) -> Bool
    sparse_dense_decision(ndims, nnz) -> Bool

Heuristic returning `true` when a symmetric linear system of dimension `ndims` with lower-triangular non-zero count `nnz`
(computed from a block `sparsity` pattern and `blocksizes` via `BlockMatrices.block_sparse_nnz`) is faster to factorize
as a sparse matrix than as a dense matrix: `64 * nnz < 25 * ndims * (ndims - 40)`.

Used by `optimvars_dispatch` to choose between a `BlockSparseMatrix` (sparse LDLᵀ) and a `BlockDenseMatrix` (dense Cholesky) linear system.
"""
sparse_dense_decision(ndims, sparsity, blocksizes) = sparse_dense_decision(ndims, block_sparse_nnz(sparsity, blocksizes))
sparse_dense_decision(ndims, nnz) = (nnz * 64) < (25 * ndims * (ndims - 40)) # Threshold nnz (for lower triangle) = 25/64 * (d^2 - 40d)
