using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra

struct DiffResidual <: NLLSModel.AbstractResidual
    i::Int
    j::Int
end
NLLSModel.ndeps(::DiffResidual) = static(2)
NLLSModel.nres(::DiffResidual) = static(1)
NLLSModel.varindices(res::DiffResidual) = SVector(res.i, res.j)
NLLSModel.getvars(res::DiffResidual, vars::Vector) = (vars[res.i]::Float64, vars[res.j]::Float64)
NLLSModel.computeresidual(::DiffResidual, a, b) = a - b
Base.eltype(::DiffResidual) = Float64

@testset "problem API" begin
    problem = Model(Float64, DiffResidual)
    for v in (1.0, 2.0, 4.0, 8.0)
        NLLSModel.addvariable!(problem, v)
    end
    NLLSModel.addcost!(problem, DiffResidual(1, 2))
    NLLSModel.addcost!(problem, DiffResidual(3, 4))
    # M4: the cost container has a concrete type
    @test isconcretetype(fieldtype(typeof(problem), :costs))
    # B8: duplicated variable indices are rejected
    @test_throws AssertionError NLLSModel.addcost!(problem, DiffResidual(2, 2))
    # B11: Boolean (non-Bit) vectors are masks of unfixed variables
    sub = NLLSModel.subproblem(problem, [true, true, false, false])
    @test NLLSModel.countcosts(NLLSModel.costnum, sub.costs) == 1
    sub = NLLSModel.subproblem(problem, BitVector([false, false, true, false]))
    @test NLLSModel.countcosts(NLLSModel.costnum, sub.costs) == 1
    @test NLLSModel.convertunfixed([2, 3], problem) == BitVector([false, true, true, false])
    @test NLLSModel.convertunfixed(Float64, problem) == trues(4)
    # M10: variables larger than 255 dimensions are rejected when added
    bigproblem = Model(Vector{Float64}, DiffResidual)
    @test_throws AssertionError NLLSModel.addvariable!(bigproblem, zeros(256))
    @test NLLSModel.addvariable!(bigproblem, zeros(255)) == 1
    # B10: scaled non-robust kernel fast path
    k = NLLSModel.Scaled(NLLSModel.NoRobust(), 2.0)
    @test NLLSModel.robustifydcost(k, 3.0) == (6.0, 2.0, 0.0)
    @test which(NLLSModel.robustifydcost, Tuple{typeof(k), Float64}).sig.parameters[2] <: NLLSModel.Scaled{<:Real, NLLSModel.NoRobust}
end

@testset "block sparse nnz with large blocks" begin
    # B9: products of UInt8 block sizes must not wrap
    pattern = sparse([true true; false true])
    sizes = NLLSModel.BlockRLE(UInt8[32, 32])
    @test NLLSModel.block_sparse_nnz(pattern, sizes) == 3 * 32 * 32
end

@testset "cholmod_symbolic_nnz" begin
    n = 20
    A = sparse(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
    fac = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(Symmetric(A, :U)))
    nz = NLLSModel.cholmod_symbolic_nnz(fac, n)
    @test nz isa Int && n <= nz <= n * (n + 1) ÷ 2
    @test isnothing(NLLSModel.cholmod_symbolic_nnz(fac, n + 1)) # mismatched size is rejected
end

module UnrollCheck
using NLLSModel
vars = () # A module-level global with the same name as the local in `count_unrolled`
function count_unrolled(vars)
    n = 0
    NLLSModel.@unroll for i in 1:3
        if i <= length(vars)
            n += 1
        end
    end
    return n
end
end
@testset "unroll ignores module globals" begin
    @test UnrollCheck.count_unrolled((1, 2)) == 2
    @test UnrollCheck.count_unrolled((1, 2, 3)) == 3
end

@testset "per-thread linear system copies are memory capped" begin
    A = NLLSModel.MultiVariateLSdense(UInt8[3, 3, 3], UInt[1, 2, 3])
    ls = NLLSModel.LinearSystem(NLLSModel.MultiVariateLSdense_x(zeros(9), UInt[1, 2, 3], [1, 4, 7, 10]), A)
    one_copy = Base.summarysize(ls.ls)
    @test NLLSModel.threadcopylimit(ls, 8) == 8
    @test NLLSModel.threadcopylimit(ls, 8, 2 * one_copy) == 3  # the original plus two copies
    @test NLLSModel.threadcopylimit(ls, 8, 0) == 1
end
