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
    bdm = BlockDenseMatrix{Float64}(H_mat, f.linsystem.x.boffsets)
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
    bsm_data = getA(f.linsystem).data
    @inbounds for (i, si) in enumerate(f.linsystem.x.sparseindices)
        H_csc.nzval[i] = bsm_data[si]
    end
    return c
end

function optimvars(problem::Model, unfixed=nothing, numthreads=StaticInt(Threads.nthreads()))
    return optimvars_dispatch(problem, convertunfixed(unfixed, problem), static(numthreads))
end

function optimvars_dispatch(problem::Model, unfixed::Integer, numthreads::StaticInt)
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

function optimvars_dispatch(problem::Model, unfixed::AbstractVector, numthreads::StaticInt)
    @assert length(problem.variables) > 0
    nblocks = sum(unfixed)
    if nblocks == 1
        return optimvars_dispatch(problem, findfirst(unfixed), numthreads)
    end
    blocksizes, blockindices = block_sizes_indices(problem.variables, unfixed, nblocks)
    len = Int(sum(blocksizes))
    if all(unfixed)
        unfixed_inds = nothing
        x0 = problem.variables
    else
        unfixed_inds = findall(unfixed)
        x0 = problem.variables[unfixed_inds]
    end
    f = MultiVarCostFunc(problem, unfixed_inds, numthreads)

    sparsity = block_sparsity(problem, unfixed)
    if len >= 40 && sparse_dense_decision(len, sparsity, blocksizes)
        bsm = BlockSparseMatrix{Float64}(sparsity, blocksizes, blocksizes)
        lssparse = MultiVariateLSsparse(bsm, blockindices, true)
        ls_shared = sharedlinearsystem(lssparse, problem, numthreads)
        fdf! = MultiVarSparseFDF(ls_shared, problem, unfixed_inds)
        return (f, fdf!, x0, lssparse.x.hessian)
    else
        lsdense = MultiVariateLSdense(blocksizes, blockindices)
        ls_shared = sharedlinearsystem(lsdense, problem, numthreads)
        fdf! = MultiVarDenseFDF(ls_shared, problem, unfixed_inds)
        return (f, fdf!, x0, nothing)
    end
end

function NLPLSModel end

NLPLSModel(problem::Model, args...; kwargs...) =
    error("NLPLSModel requires the NLPModels package. Run `using NLPModels` first.")

function NLLSNLSModel end

NLLSNLSModel(problem::Model, args...; kwargs...) =
    error("NLLSNLSModel requires the NLPModels package. Run `using NLPModels` first.")
