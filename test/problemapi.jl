using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra

include("fixtures/loader.jl")
fixture("problemapi")

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
    @test_throws ArgumentError NLLSModel.addcost!(problem, DiffResidual(2, 2))
    # B11: Boolean (non-Bit) vectors are masks of unfixed variables
    sub = NLLSModel.subproblem(problem, [true, true, false, false])
    @test NLLSModel.countcosts(NLLSModel.costnum, sub.costs) == 1
    sub = NLLSModel.subproblem(problem, BitVector([false, false, true, false]))
    @test NLLSModel.countcosts(NLLSModel.costnum, sub.costs) == 1
    @test NLLSModel.convertunfixed([2, 3], problem) == BitVector([false, true, true, false])
    @test NLLSModel.convertunfixed(Float64, problem) == trues(4)
    # M10: any variable size can be added; only a block of a block linear system is limited, by BlockLinearSystems (see test/bigvariable.jl)
    bigproblem = Model(Vector{Float64}, DiffResidual)
    @test NLLSModel.addvariable!(bigproblem, zeros(256)) == 1
    # B10: scaled non-robust kernel fast path
    k = NLLSModel.Scaled(NLLSModel.NoRobust(), 2.0)
    @test NLLSModel.robustifydcost(k, 3.0) == (6.0, 2.0, 0.0)
    @test which(NLLSModel.robustifydcost, Tuple{typeof(k), Float64}).sig.parameters[2] <: NLLSModel.Scaled{<:Real, NLLSModel.NoRobust}
end

@testset "Model constructors and show" begin
    problem = Model([1.0, 2.0, 4.0], DiffResidual)
    @test problem isa Model{Float64, DiffResidual}
    addcost!(problem, DiffResidual(1, 2))
    addcost!(problem, DiffResidual(2, 3))
    @test sprint(show, problem) == "Model with 3 variable blocks of total dimension 3,\n             and 2 residual blocks of total dimension 2."
    fromcosts = Model([1.0, 2.0], NLLSModel.CostStruct([DiffResidual(1, 2)]))
    @test NLLSModel.cost(fromcosts) ≈ 0.5
    # A model with no cost type; the call is made at run time, as a constant one is folded by the compiler
    @test NLLSModel.tupelize(Base.inferencebarrier(Union{})) === Tuple{}
    # Without NLPModels, the wrappers say what to load (here called past the extension's own methods)
    @test_throws "requires the NLPModels package" invoke(NLPLSModel, Tuple{Model, Vararg{Any}}, problem)
    @test_throws "requires the NLPModels package" invoke(NLLSNLSModel, Tuple{Model, Vararg{Any}}, problem)
end

@testset "SimpleError3 and SimpleError4" begin
    model = Model(V2, Union{SimpleError3{2, Float64, V2, V2, V2}, SimpleError4{2, Float64, V2, V2, V2, V2}})
    for v in (V2(1, 2), V2(-1, 0.5), V2(3, 1), V2(0, -2))
        addvariable!(model, v)
    end
    a, b, c, d = model.variables
    m3, m4 = V2(0.5, 0.5), V2(1, -1)
    addcost!(model, SimpleError3{V2, V2, V2}(m3, 1, 2, 3))
    addcost!(model, SimpleError4{V2, V2, V2, V2}(m4, 1, 2, 3, 4))
    r3, r4 = a + 2 * b - c - m3, a + b + c + d - m4
    @test NLLSModel.cost(model) ≈ 0.5 * (sum(abs2, r3) + sum(abs2, r4))
    _, fdf!, x0, _ = optimvars(model; numthreads=static(1), storage=:dense, reduction=:none)
    g, H = zeros(8), zeros(8, 8)
    @test fdf!(g, H, x0) ≈ NLLSModel.cost(model)
    @test g ≈ vcat(r3 + r4, 2 * r3 + r4, -r3 + r4, r4)
end

@testset "non-finite checks of other element types" begin
    # Run-time values, so that the methods are called rather than folded
    @test NLLSModel.is_nonfinite(Base.inferencebarrier(NaN32)) && !NLLSModel.is_nonfinite(Base.inferencebarrier(1.0f0))
    @test NLLSModel.safe_sqnorm(Base.inferencebarrier(-3.0f0)) === 9.0
end

@testset "subproblem: each form of unfixed selects the costs touching those variables (AN2)" begin
    # Cost k does not touch variable k, so variable indices read as cost indices select other costs. Variables 1 and 3 are
    # of another type, for the type form.
    problem = Model(Union{Float64, Float32}, EitherDiff)
    for v in (1.0f0, 2.0, 4.0f0, 8.0, 16.0)
        NLLSModel.addvariable!(problem, v)
    end
    for (i, j) in ((2, 3), (3, 4), (4, 5), (5, 2))
        NLLSModel.addcost!(problem, EitherDiff(i, j))
    end
    pairs(model) = sort([(c.i, c.j) for costs in values(model.costs) for c in costs])
    touching(mask) = [(i, j) for (i, j) in ((2, 3), (3, 4), (4, 5), (5, 2)) if mask[i] || mask[j]]
    for (name, unfixed, mask) in (
            ("a list of indices", [1, 3], Bool[1, 0, 1, 0, 0]),
            ("a list of one index beyond the cost count", [5], Bool[0, 0, 0, 0, 1]),
            ("an index", 4, Bool[0, 0, 0, 1, 0]),
            ("a type", Float32, Bool[1, 0, 1, 0, 0]),
            ("a Boolean vector", Bool[0, 1, 0, 0, 0], Bool[0, 1, 0, 0, 0]),
            ("a BitVector", BitVector([0, 0, 0, 0, 1]), Bool[0, 0, 0, 0, 1]),
            ("nothing", nothing, Bool[1, 1, 1, 1, 1]))
        @testset "$name" begin
            sub = NLLSModel.subproblem(problem, unfixed)
            @test pairs(sub) == touching(mask)
            @test pairs(sub) == pairs(NLLSModel.subproblem(problem, mask))
            @test sub.variables === problem.variables
        end
    end
    # subproblem! takes cost indices, sorted and in range
    to = NLLSModel.subproblem(problem, nothing)
    @test pairs(NLLSModel.subproblem!(to, problem, [1, 3])) == [(2, 3), (4, 5)]
    @test pairs(NLLSModel.subproblem!(to, problem, Int[])) == []
    for bad in ([3, 1], [0, 2], [2, 5], [1, 1, 2, 9])
        @test_throws ArgumentError NLLSModel.subproblem!(to, problem, bad)
    end
end

@testset "input checks throw ArgumentError, naming the type (AN8)" begin
    msg(f) = try f(); "no error" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError: $(typeof(e))" end
    for (K, text) in ((:ndeps, "ndeps()"), (:varindices, "varindices()"), (:getvars, "getvars"), (:nres, "nres()"))
        problem = Model(Float64, BadRes{K})
        NLLSModel.addvariable!(problem, 1.0)
        m = msg(() -> NLLSModel.addcost!(problem, BadRes{K}()))
        @test occursin(text, m) && occursin("BadRes{:$K}", m)
    end
    problem = Model(Float64, DiffResidual)
    NLLSModel.addvariable!(problem, 1.0)
    NLLSModel.addvariable!(problem, 2.0)
    m = msg(() -> NLLSModel.addcost!(problem, DiffResidual(2, 2)))
    @test occursin("same variable more than once", m) && occursin("DiffResidual", m)
    problem = Model(Float64, BadAdaptive)
    NLLSModel.addvariable!(problem, 1.0)
    m = msg(() -> NLLSModel.addcost!(problem, BadAdaptive()))
    @test occursin("adaptive robustifier", m) && occursin("BadAdaptive", m)
    # A variable with no dimensions
    problem = Model(Vector{Float64}, DiffResidual)
    m = msg(() -> NLLSModel.addvariable!(problem, Float64[]))
    @test occursin("nvars()", m) && occursin("Vector{Float64}", m)
    # A model with no variables
    @test occursin("no variables", msg(() -> optimvars(Model(Float64, DiffResidual))))
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

@testset "unroll ignores module globals" begin
    @test UnrollCheck.count_unrolled((1, 2)) == 2
    @test UnrollCheck.count_unrolled((1, 2, 3)) == 3
end

@testset "unroll folds constant conditions" begin
    @test UnrollCheck.unrolled_branches() == [1, 2, -3, 0]
    ex = macroexpand(UnrollCheck, quote
        NLLSModel.@unroll for i in 1:2
            if i == 1
                a
            end
        end
    end)
    @test !occursin("if", string(ex)) # Both conditions folded: `a`, then `nothing`
    # Anything but a plain for-loop is rejected
    notfor = (quote
        NLLSModel.@unroll while true
        end
    end, quote
        NLLSModel.@unroll for (i, j) in pairs(1:2)
        end
    end)
    for ex in notfor
        @test_throws "must be a for-loop" macroexpand(UnrollCheck, ex)
    end
end

@testset "per-thread linear system copies are memory capped" begin
    A = NLLSModel.MultiVariateLSdense([3, 3, 3], UInt[1, 2, 3])
    ls = NLLSModel.LinearSystem(NLLSModel.MultiVariateLSdense_x(zeros(9), UInt[1, 2, 3], [1, 4, 7, 10]), A)
    one_copy = Base.summarysize(ls.ls)
    @test NLLSModel.threadcopylimit(ls, 8) == 8
    @test NLLSModel.threadcopylimit(ls, 8, 2 * one_copy) == 3  # the original plus two copies
    @test NLLSModel.threadcopylimit(ls, 8, 0) == 1
end
