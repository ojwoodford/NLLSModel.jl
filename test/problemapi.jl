using NLLSModel, Test, Static, StaticArrays, SparseArrays

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
    # B8: duplicated variable indices are rejected
    @test_throws AssertionError NLLSModel.addcost!(problem, DiffResidual(2, 2))
    # B11: Boolean (non-Bit) vectors are masks of unfixed variables
    sub = NLLSModel.subproblem(problem, [true, true, false, false])
    @test NLLSModel.countcosts(NLLSModel.costnum, sub.costs) == 1
    sub = NLLSModel.subproblem(problem, BitVector([false, false, true, false]))
    @test NLLSModel.countcosts(NLLSModel.costnum, sub.costs) == 1
    @test NLLSModel.convertunfixed([2, 3], problem) == BitVector([false, true, true, false])
    @test NLLSModel.convertunfixed(Float64, problem) == trues(4)
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
