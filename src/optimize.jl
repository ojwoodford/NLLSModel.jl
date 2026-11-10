using Static, StaticArrays, SparseArrays, LinearAlgebra

# Conversions for unfixed
convertunfixed(::Nothing, problem) = trues(length(problem.variables))
convertunfixed(unfixed::DataType, problem) = isa.(problem.variables, unfixed)
convertunfixed(unfixed::Integer, problem) = unfixed
convertunfixed(unfixed, problem) = unfixed

# Single-variable objective and gradient/Hessian functors
struct SingleVarCostFunc{P, U, NT}
    problem::P
    unfixed::U
    numthreads::NT
end
@inline function (f::SingleVarCostFunc)(x)
    @inbounds f.problem.variables[f.unfixed] = x
    return cost(f.problem.variables, f.problem.costs, f.numthreads)
end

struct SingleVarCostGradHess{LS, P, U}
    linsystem::LS
    problem::P
    unfixed::U
end
@inline makels(::UniVariateLSstatic_ls{N, N2}, H, g) where {N, N2} = UniVariateLSstatic_ls{N, N2}(H, g)
@inline makels(::UniVariateLSdynamic_ls, H, g) = UniVariateLSdynamic_ls(H, g)

@inline function (f::SingleVarCostGradHess)(g, H, x)
    @inbounds f.problem.variables[f.unfixed] = x
    if f.linsystem isa LinearSystemShared
        if H isa typeof(f.linsystem.ls[1].A) && g isa typeof(f.linsystem.ls[1].b)
            ls1 = makels(f.linsystem.ls[1], H, g)
            target = replace_first_ls(f.linsystem, ls1)
            return costgradhess!(target, f.problem.variables, f.problem.costs)
        else
            c = costgradhess!(f.linsystem, f.problem.variables, f.problem.costs)
            copyto!(g, getb(f.linsystem))
            copyto!(H, getA(f.linsystem))
            return c
        end
    else
        if H isa typeof(f.linsystem.ls.A) && g isa typeof(f.linsystem.ls.b)
            target_ls = LinearSystem(f.linsystem.x, makels(f.linsystem.ls, H, g))
            zero!(target_ls)
            return costgradhess_nozero!(target_ls, f.problem.variables, f.problem.costs)
        else
            c = costgradhess!(f.linsystem, f.problem.variables, f.problem.costs)
            copyto!(g, getb(f.linsystem))
            copyto!(H, getA(f.linsystem))
            return c
        end
    end
end

# Multi-variable objective and gradient/Hessian functors
@inline sync_unfixed_vars!(::Model, ::Nothing, vars) = vars
@inline function sync_unfixed_vars!(problem::Model, unfixed_inds::Vector{Int}, vars)
    @inbounds for (k, idx) in enumerate(unfixed_inds)
        problem.variables[idx] = vars[k]
    end
    return problem.variables
end

struct MultiVarCostFunc{P, U, NT}
    problem::P
    unfixed_inds::U
    numthreads::NT
end
@inline function (f::MultiVarCostFunc)(vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    return cost(full_vars, f.problem.costs, f.numthreads)
end

struct MultiVarDenseFDF{LS, P, U}
    linsystem::LS
    problem::P
    unfixed_inds::U
end
@inline function (f::MultiVarDenseFDF)(g, H_mat::Matrix{Float64}, vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    orig_A = getA(f.linsystem)
    bdm = DenseBlockLinearSystem{Float64, Matrix{Float64}}(H_mat, orig_A.rowblockoffsets, orig_A.columnblockoffsets, orig_A.rowblocksizes, orig_A.columnblocksizes)
    if f.linsystem isa LinearSystemShared
        ls1 = MultiVariateLSdense_ls(bdm, g)
        target = replace_first_ls(f.linsystem, ls1)
        c = costgradhess!(target, full_vars, f.problem.costs)
    else
        target_ls = LinearSystem(f.linsystem.x, MultiVariateLSdense_ls(bdm, g))
        zero!(target_ls)
        c = costgradhess_nozero!(target_ls, full_vars, f.problem.costs)
    end
    symmetrifyfull!(bdm)
    return c
end

struct MultiVarSparseFDF{LS, P, U}
    linsystem::LS
    problem::P
    unfixed_inds::U
end
@inline function (f::MultiVarSparseFDF)(g, H_csc::SparseMatrixCSC{Float64, Int}, vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    if f.linsystem isa LinearSystemShared
        ls1 = MultiVariateLSsparse_ls(f.linsystem.ls[1].A, g)
        target = replace_first_ls(f.linsystem, ls1)
        c = costgradhess!(target, full_vars, f.problem.costs)
    else
        target_ls = LinearSystem(f.linsystem.x, MultiVariateLSsparse_ls(f.linsystem.ls.A, g))
        zero!(target_ls)
        c = costgradhess_nozero!(target_ls, full_vars, f.problem.costs)
    end
    copyto!(H_csc, getA(f.linsystem), f.linsystem.x.sparseindices)
    return c
end

struct MultiVarSchurFDF{LS, P, U}
    linsystem::LS
    problem::P
    unfixed_inds::U
end
@inline function (f::MultiVarSchurFDF)(g, H_schur::SchurBlockMatrix, vars)
    full_vars = sync_unfixed_vars!(f.problem, f.unfixed_inds, vars)
    if f.linsystem isa LinearSystemShared
        ls1 = MultiVariateLSSchur_ls(H_schur, g)
        target = replace_first_ls(f.linsystem, ls1)
        c = costgradhess!(target, full_vars, f.problem.costs)
    else
        target_ls = LinearSystem(f.linsystem.x, MultiVariateLSSchur_ls(H_schur, g))
        zero!(target_ls)
        c = costgradhess_nozero!(target_ls, full_vars, f.problem.costs)
    end
    return c
end

function cholmod_symbolic_nnz(fac::SparseArrays.CHOLMOD.Factor)
    s = unsafe_load(fac.ptr)
    if s.ColCount != C_NULL
        colcount = unsafe_wrap(Array, Ptr{Int64}(s.ColCount), s.n)
        return sum(colcount)
    end
    return 0
end

function optimvars(problem::Model, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()); linsolver=nothing)
    return optimvars_dispatch(problem, convertunfixed(unfixed, problem), static(numthreads), linsolver)
end

function optimvars_dispatch(problem::Model, unfixed::Integer, numthreads::StaticInt, linsolver=nothing)
    x0 = problem.variables[unfixed]
    varlen = nvars(x0)
    if dynamic(is_static(varlen)) && varlen <= 16
        ls = LinearSystem{UniVariateLSstatic_x{dynamic(varlen)}, UniVariateLSstatic_ls{dynamic(varlen), dynamic(varlen * varlen)}}((unfixed,), ())
    else
        ls = LinearSystem{UniVariateLSdynamic_x, UniVariateLSdynamic_ls}((unfixed, dynamic(varlen)), (dynamic(varlen),))
    end
    ls_shared = sharedlinearsystem(ls, problem, numthreads)
    f = SingleVarCostFunc(problem, unfixed, numthreads)
    fdf! = SingleVarCostGradHess(ls_shared, problem, unfixed)
    return (f, fdf!, x0, nothing)
end

function optimvars_dispatch(problem::Model, unfixed::AbstractVector, numthreads::StaticInt, linsolver=nothing)
    @assert length(problem.variables) > 0
    nblocks = sum(unfixed)
    if nblocks == 1
        return optimvars_dispatch(problem, findfirst(unfixed), numthreads, linsolver)
    end

    linsolver_sym = isnothing(linsolver) ? nothing : Symbol(linsolver)
    schur_requested = linsolver_sym in (:schur, :dense_schur, :sparse_schur)
    has_elims = any(v -> dynamic(OptimizationVariableInterface.iseliminable(typeof(v))), problem.variables)

    if schur_requested || ((isnothing(linsolver_sym) || linsolver_sym === :auto) && has_elims)
        partition = partition_schur_problem(problem, unfixed, linsolver_sym)
        if !isnothing(partition)
            structure = SchurStructure(partition.c_dims, partition.p_dims, partition.observations; nchunks=dynamic(numthreads))
            
            # Decision Matrix: Dense vs. Sparse
            is_dense = if linsolver_sym === :dense_schur
                true
            elseif linsolver_sym === :sparse_schur
                false
            else
                if structure.nc <= 2000
                    covis = build_schur_covisibility(structure)
                    chol_fac = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Symmetric(covis.S_sparse, :U)))
                    cholmod_symbolic_nnz(chol_fac) >= 0.40 * 0.5 * structure.nc^2
                else
                    false
                end
            end

            total_c_size = sum(Int(d)^2 for d in structure.p_dims; init=0)
            total_e_size = isempty(structure.cpoffsets) ? 0 : (structure.cpoffsets[end] + Int(structure.c_dims[structure.cpblock[end]]) * Int(structure.p_dims[end]) - 1)
            C_data = zeros(Float64, total_c_size)
            E_data = zeros(Float64, total_e_size)

            H = if is_dense
                B_dense = zeros(Float64, structure.nc, structure.nc)
                SchurBlockMatrix(structure, B_dense, C_data, E_data)
            else
                covis = build_schur_covisibility(structure)
                B_sys = SparseBlockLinearSystem{Float64}(TransposedPattern(covis.block_covis), structure.c_dims, structure.c_dims)
                SchurBlockMatrix(structure, B_sys, C_data, E_data)
            end

            nblocks_tot = partition.ncblocks + partition.npblocks
            blockindices = zeros(UInt, length(problem.variables))
            for (b, var_idx) in enumerate(partition.unfixed_inds)
                blockindices[var_idx] = b
            end

            boffsets = Vector{Int}(undef, nblocks_tot + 1)
            for b in 1:partition.ncblocks
                boffsets[b] = structure.boffsets_c[b]
            end
            for l in 1:partition.npblocks
                boffsets[partition.ncblocks + l] = structure.nc + structure.boffsets_p[l]
            end
            boffsets[end] = structure.nc + sum(Int, structure.p_dims; init=0) + 1

            tot_len = structure.nc + sum(Int, structure.p_dims; init=0)
            x_vec = zeros(Float64, tot_len)
            b_vec = zeros(Float64, tot_len)

            ls_x = MultiVariateLSSchur_x(x_vec, blockindices, boffsets, structure, partition.unfixed_inds)
            ls_val = MultiVariateLSSchur_ls(H, b_vec)
            lsschur = LinearSystem(ls_x, ls_val)
            ls_shared = sharedlinearsystem(lsschur, problem, numthreads)

            f = MultiVarCostFunc(problem, partition.unfixed_inds, numthreads)
            fdf! = MultiVarSchurFDF(ls_shared, problem, partition.unfixed_inds)
            x0 = problem.variables[partition.unfixed_inds]
            return (f, fdf!, x0, H)
        end
    end

    blocksizes, blockindices, unfixed_inds, is_natural = block_sizes_indices(problem.variables, unfixed, nblocks)
    len = Int(sum(blocksizes))
    if all(unfixed) && is_natural
        unfixed_sync = nothing
        x0 = problem.variables
    else
        unfixed_sync = unfixed_inds
        x0 = problem.variables[unfixed_inds]
    end
    f = MultiVarCostFunc(problem, unfixed_sync, numthreads)

    sparsity = block_sparsity(problem, unfixed_inds)
    if len >= 40 && sparse_dense_decision(len, sparsity, blocksizes)
        bsm = SparseBlockLinearSystem{Float64}(TransposedPattern(sparsity), blocksizes, blocksizes)
        lssparse = MultiVariateLSsparse(bsm, blockindices, true)
        ls_shared = sharedlinearsystem(lssparse, problem, numthreads)
        fdf! = MultiVarSparseFDF(ls_shared, problem, unfixed_sync)
        return (f, fdf!, x0, lssparse.x.hessian)
    else
        lsdense = MultiVariateLSdense(blocksizes, blockindices)
        ls_shared = sharedlinearsystem(lsdense, problem, numthreads)
        fdf! = MultiVarDenseFDF(ls_shared, problem, unfixed_sync)
        return (f, fdf!, x0, nothing)
    end
end

function NLPLSModel end

NLPLSModel(problem::Model, args...; kwargs...) =
    error("NLPLSModel requires the NLPModels package. Run `using NLPModels` first.")

function NLLSNLSModel end

NLLSNLSModel(problem::Model, args...; kwargs...) =
    error("NLLSNLSModel requires the NLPModels package. Run `using NLPModels` first.")
