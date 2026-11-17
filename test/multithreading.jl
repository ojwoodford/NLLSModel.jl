using Test, Static, StaticArrays, SparseArrays, LinearAlgebra, NLLSModel

include("fixtures/multithreading.jl")

@testset "multithreading.jl" begin
    # chunk_indices: chunks are contiguous, cover 1:len exactly, and differ in size by at most 1
    for len in (0, 1, 5, 16, 17, 1001), nchunks in (1, 2, 3, 4, 7, 20)
        chunks = [NLLSModel.chunk_indices(len, nchunks, id) for id in 1:nchunks]
        @test reduce(vcat, collect.(chunks); init=Int[]) == collect(1:len)
        @test maximum(length, chunks) - minimum(length, chunks) <= 1
        @test isempty(NLLSModel.chunk_indices(len, nchunks, nchunks + 1))
    end
    @test NLLSModel.chunk_indices(10, 3, 1) == 1:4
    @test NLLSModel.chunk_indices(10, 3, 2) == 5:7
    @test NLLSModel.chunk_indices(10, 3, 3) == 8:10

    # ithchunk
    coll = collect(11:20)
    @test NLLSModel.ithchunk(3, 2, coll) == [15, 16, 17]
    @test NLLSModel.ithchunk(3, 2, coll) isa SubArray
    @test isempty(NLLSModel.ithchunk(3, 4, coll))

    # threadargs: an isbits stand-in for the arguments of a `@batch` loop, which fromthreadargs turns back into the same
    # objects. A field of abstract type is passed by the address of its box; one of an isbits union as it is.
    @testset "threadargs" begin
        mv, vv = MVector(1.0, 2.0), [zeros(2), ones(3)]
        nested = (SVector(mv, mv), vv, (a=1, b=vv), 3.0)
        a = NLLSModel.threadargs(nested)
        @test isbits(a)
        back = NLLSModel.fromthreadargs(a)
        @test back === nested
        @test back[1][1] === mv && back[2] === vv
        @test allocs_warm(NLLSModel.threadargs, nested) == 0
        @test allocs_warm(NLLSModel.fromthreadargs, a) == 0
        abstractfield = (Any[1], Ref{Any}(2.0), Some{Any}(vv), Some{Any}(nothing), Some{Any}((1, mv)), Some{Union{Nothing, Vector{Vector{Float64}}}}(vv))
        b = NLLSModel.threadargs(abstractfield)
        @test isbits(b)
        @test NLLSModel.fromthreadargs(b) === abstractfield
        @test allocs_warm(NLLSModel.threadargs, abstractfield) == 0
        bitsunion = (Some{Union{Nothing, Int}}(3), mv)
        c = NLLSModel.threadargs(bitsunion)
        @test !isbits(c) # Stored inline, so passed as it is
        @test NLLSModel.fromthreadargs(c) === bitsunion
        @test NLLSModel.fromthreadargs(NLLSModel.threadargs(5)) === 5
    end

    # cost_range
    vars_cr = [2.0]
    costs_cr = [MTTestScalarRes(k * 0.1, 1) for k in 1:100]
    serial = sum(c -> NLLSModel.computecost(vars_cr, c), costs_cr)
    @test NLLSModel.cost_range(vars_cr, costs_cr, 1:100) ≈ serial
    @test NLLSModel.cost_range(vars_cr, costs_cr, 1:0) == 0.0
    @test NLLSModel.cost_range(vars_cr, costs_cr, 1:50) + NLLSModel.cost_range(vars_cr, costs_cr, 51:100) ≈ serial

    # Threaded cost matches serial cost for any thread count, above and below the threading threshold,
    # and for both Union and Tuple CostStructs
    for ncosts in (100, 60000)
        prob_union = Model{Float64, Union{MTTestScalarRes, MTTestPairRes}}()
        prob_tuple = Model{Float64, Tuple{MTTestScalarRes, MTTestPairRes}}()
        for p in (prob_union, prob_tuple)
            addvariable!(p, 1.5)
            addvariable!(p, -0.5)
            for k in 1:ncosts
                addcost!(p, MTTestScalarRes(k * 1.e-4, 1))
                addcost!(p, MTTestPairRes(k * 1.e-4, 1, 2))
            end
        end
        ref = NLLSModel.cost(prob_union.variables, prob_union.costs, static(1))
        @test ref ≈ sum(c -> NLLSModel.computecost(prob_union.variables, c), get(prob_union.costs, MTTestScalarRes)) +
                    sum(c -> NLLSModel.computecost(prob_union.variables, c), get(prob_union.costs, MTTestPairRes))
        for n in (1, 2, 3, 4, 7, Threads.nthreads() + 1)
            @test NLLSModel.cost(prob_union.variables, prob_union.costs, static(n)) ≈ ref
            @test NLLSModel.cost(prob_tuple.variables, prob_tuple.costs, static(n)) ≈ ref
            @test NLLSModel.cost(prob_union, n) ≈ ref
        end
    end

    # Cost sums
    v = rand(Int(1e6))
    s1 = sum(myfun, v)
    s2 = sum(NLLSModel.bindleadingargs(computecost, static(1), Vector{Int}()), v)
    @test isapprox(s1, s2; rtol=1.e-13)
    s3 = sum(NLLSModel.bindleadingargs(computecost, static(Threads.nthreads()), Vector{Int}()), v)
    @test isapprox(s1, s3; rtol=1.e-13)

    # Multi-threaded costgradhess! vs single-threaded
    # 1. UniVariateLSstatic
    prob_static = Model(Float64, MTTestScalarRes)
    addvariable!(prob_static, 2.0)
    for k in 1:120
        addcost!(prob_static, MTTestScalarRes(k * 0.1, 1))
    end
    ls_static_single = NLLSModel.LinearSystem{NLLSModel.UniVariateLSstatic_x{1}, NLLSModel.UniVariateLSstatic_ls{1, 1}}((1,), ())
    ls_static_shared = NLLSModel.sharedlinearsystem(ls_static_single, static(4))
    c_s = NLLSModel.costgradhess!(ls_static_single, prob_static.variables, prob_static.costs)
    c_m = NLLSModel.costgradhess!(ls_static_shared, prob_static.variables, prob_static.costs)
    @test c_s ≈ c_m
    @test NLLSModel.getgrad(ls_static_single) ≈ NLLSModel.getgrad(ls_static_shared)
    @test NLLSModel.gethessian(ls_static_single) ≈ NLLSModel.gethessian(ls_static_shared)
    c_cost_s = NLLSModel.cost(prob_static.variables, prob_static.costs, static(1))
    c_cost_m = NLLSModel.cost(prob_static.variables, prob_static.costs, static(4))
    @test c_cost_s ≈ c_cost_m
    @test c_cost_s ≈ c_s
    @test NLLSModel.cost(prob_static.variables, prob_static.costs) ≈ c_cost_s
    @test NLLSModel.cost(prob_static.variables, prob_static.costs, 2) ≈ c_cost_s

    # LinearSystemShared with N == 1
    ls_static_shared_1 = NLLSModel.LinearSystemShared(ls_static_single.x, SVector{1}(ls_static_single.ls), MVector{1, Float64}(undef))
    @test NLLSModel.costgradhess!(ls_static_shared_1, prob_static.variables, prob_static.costs) ≈ c_s

    # Threaded costgradhess! with concrete DataType and Union CostStructs
    prob_static_typed = Model{Float64, MTTestScalarRes}()
    addvariable!(prob_static_typed, 2.0)
    for k in 1:120
        addcost!(prob_static_typed, MTTestScalarRes(k * 0.1, 1))
    end
    c_st = NLLSModel.costgradhess!(ls_static_single, prob_static_typed.variables, prob_static_typed.costs)
    c_mt = NLLSModel.costgradhess!(ls_static_shared, prob_static_typed.variables, prob_static_typed.costs)
    @test c_st ≈ c_mt

    prob_static_union = Model{Float64, Union{MTTestScalarRes, MTTestPairRes}}()
    addvariable!(prob_static_union, 2.0)
    addvariable!(prob_static_union, 1.0)
    for k in 1:60
        addcost!(prob_static_union, MTTestScalarRes(k * 0.1, 1))
        addcost!(prob_static_union, MTTestPairRes(k * 0.1, 1, 2))
    end
    c_su = NLLSModel.costgradhess!(ls_static_single, prob_static_union.variables, prob_static_union.costs)
    c_mu = NLLSModel.costgradhess!(ls_static_shared, prob_static_union.variables, prob_static_union.costs)
    @test c_su ≈ c_mu

    # Verify ephemeral stack wrapper construction has 0 allocations
    ls_ephemeral = NLLSModel.LinearSystem(ls_static_shared, 1)
    @test isa(ls_ephemeral, NLLSModel.LinearSystem)
    allocs = alloc_ephemeral(ls_static_shared, 1)
    @test allocs == 0

    # Additional linearsystem coverage
    @test NLLSModel.copy_structure(ls_static_single) isa NLLSModel.LinearSystem
    @test NLLSModel.sharedlinearsystem(ls_static_shared, static(4)) === ls_static_shared
    @test NLLSModel.sharedlinearsystem(ls_static_single, 4) isa NLLSModel.LinearSystemShared
    ls_ep2 = NLLSModel.LinearSystem(ls_static_shared, 2)
    NLLSModel.add!(ls_ephemeral, ls_ep2)
    @test NLLSModel.getgrad(ls_static_single.ls) === ls_static_single.ls.b
    @test NLLSModel.gethessian(ls_static_single.ls) === ls_static_single.ls.A
    @test NLLSModel.getA(ls_static_single.ls) === ls_static_single.ls.A
    @test NLLSModel.getb(ls_static_single.ls) === ls_static_single.ls.b
    @test NLLSModel.getx(ls_static_single.x) === ls_static_single.x.x
    lss_direct = NLLSModel.LinearSystemShared{NLLSModel.UniVariateLSstatic_x{1}, NLLSModel.UniVariateLSstatic_ls{1, 1}, 2}((1,), ())
    @test lss_direct isa NLLSModel.LinearSystemShared
    @test NLLSModel.updateunfixed(ls_static_shared, 1) isa NLLSModel.LinearSystemShared

    # 2. UniVariateLSdynamic
    dim = 20
    prob_dyn = Model(Vector{Float64}, MTTestDynamicRes)
    addvariable!(prob_dyn, zeros(dim))
    for k in 1:120
        addcost!(prob_dyn, MTTestDynamicRes(1.0, randn(dim)))
    end
    ls_dyn_single = NLLSModel.LinearSystem{NLLSModel.UniVariateLSdynamic_x, NLLSModel.UniVariateLSdynamic_ls}((1, dim), (dim,))
    ls_dyn_shared = NLLSModel.sharedlinearsystem(ls_dyn_single, static(4))
    c_ds = NLLSModel.costgradhess!(ls_dyn_single, prob_dyn.variables, prob_dyn.costs)
    c_dm = NLLSModel.costgradhess!(ls_dyn_shared, prob_dyn.variables, prob_dyn.costs)
    @test c_ds ≈ c_dm
    @test NLLSModel.getgrad(ls_dyn_single) ≈ NLLSModel.getgrad(ls_dyn_shared)
    @test NLLSModel.gethessian(ls_dyn_single) ≈ NLLSModel.gethessian(ls_dyn_shared)

    # 3. MultiVariateLSdense
    prob_dense = Model(Float64, Union{MTTestPairRes, MTTestScalarRes})
    addvariable!(prob_dense, 1.0)
    addvariable!(prob_dense, 2.0)
    for k in 1:120
        addcost!(prob_dense, MTTestPairRes(0.5, 1, 2))
        addcost!(prob_dense, MTTestScalarRes(1.0, 1))
    end
    blocksizes, blockindices = NLLSModel.block_sizes_indices(prob_dense.variables, trues(2))
    ls_dense_single = NLLSModel.MultiVariateLSdense(blocksizes, blockindices)
    ls_dense_shared = NLLSModel.sharedlinearsystem(ls_dense_single, static(4))
    c_dens_s = NLLSModel.costgradhess!(ls_dense_single, prob_dense.variables, prob_dense.costs)
    c_dens_m = NLLSModel.costgradhess!(ls_dense_shared, prob_dense.variables, prob_dense.costs)
    @test c_dens_s ≈ c_dens_m
    @test NLLSModel.getgrad(ls_dense_single) ≈ NLLSModel.getgrad(ls_dense_shared)
    @test NLLSModel.gethessian(ls_dense_single) ≈ NLLSModel.gethessian(ls_dense_shared)
    ls_dense_reconstructed = NLLSModel.MultiVariateLSdense(ls_dense_single.ls.A, ls_dense_single.ls.b, ls_dense_single.x.x, ls_dense_single.x.blockindices, ls_dense_single.x.boffsets)
    @test ls_dense_reconstructed isa NLLSModel.LinearSystem

    # 4. MultiVariateLSsparse
    prob_sparse = Model(Float64, MTTestPairRes)
    nvars = 40
    for i in 1:nvars
        addvariable!(prob_sparse, Float64(i))
    end
    for i in 1:nvars-1
        addcost!(prob_sparse, MTTestPairRes(1.0, i, i+1))
    end
    b_sizes, b_indices = NLLSModel.block_sizes_indices(prob_sparse.variables, trues(nvars))
    sparsity = NLLSModel.block_sparsity(prob_sparse, trues(nvars))
    bsm = NLLSModel.SparseBlockLinearSystem{Float64}(NLLSModel.TransposedPattern(sparsity), b_sizes, b_sizes)
    ls_sparse_single = NLLSModel.MultiVariateLSsparse(bsm, b_indices, true)
    ls_sparse_shared = NLLSModel.sharedlinearsystem(ls_sparse_single, static(4))
    c_sp_s = NLLSModel.costgradhess!(ls_sparse_single, prob_sparse.variables, prob_sparse.costs)
    c_sp_m = NLLSModel.costgradhess!(ls_sparse_shared, prob_sparse.variables, prob_sparse.costs)
    @test c_sp_s ≈ c_sp_m
    @test NLLSModel.getgrad(ls_sparse_single) ≈ NLLSModel.getgrad(ls_sparse_shared)
    @test NLLSModel.gethessian(ls_sparse_single) ≈ NLLSModel.gethessian(ls_sparse_shared)
    ls_sparse_reconstructed = NLLSModel.MultiVariateLSsparse(ls_sparse_single.ls.A, ls_sparse_single.ls.b, ls_sparse_single.x.x, ls_sparse_single.x.blockindices, ls_sparse_single.x.boffsets, ls_sparse_single.x.hessian, ls_sparse_single.x.sparseindices)
    @test ls_sparse_reconstructed isa NLLSModel.LinearSystem
    ls_sparse_noh = NLLSModel.MultiVariateLSsparse(bsm, b_indices, false)
    @test ls_sparse_noh isa NLLSModel.LinearSystem

    # optimvars single-threaded vs multi-threaded equivalence
    for prob in (prob_static, prob_dyn, prob_dense, prob_sparse)
        f_s, fdf_s!, x0_s, H_s = optimvars(deepcopy(prob); numthreads=static(1))
        f_m, fdf_m!, x0_m, H_m = optimvars(deepcopy(prob); numthreads=static(4))
        @test f_s(x0_s) ≈ f_m(x0_m)
        dim_x = dynamic(NLLSModel.nvars(x0_s))
        g_s = zeros(dim_x)
        g_m = zeros(dim_x)
        H_buf_s = isnothing(H_s) ? zeros(dim_x, dim_x) : copy(H_s)
        H_buf_m = isnothing(H_m) ? zeros(dim_x, dim_x) : copy(H_m)
        @test fdf_s!(g_s, H_buf_s, x0_s) ≈ fdf_m!(g_m, H_buf_m, x0_m)
        @test g_s ≈ g_m
        @test Matrix(H_buf_s) ≈ Matrix(H_buf_m)
    end

    # Threshold-based decision: a linearization estimate below twice the per-thread floor stays single-threaded; at
    # exactly twice the default floor it shares two copies
    @test isa(NLLSModel.sharedlinearsystem(ls_static_single, NLLSModel.workestimates(prob_static, nothing).normal, static(4), NLLSModel.costsplit(prob_static.costs)), NLLSModel.LinearSystem)
    prob_large = Model(Float64, MTTestScalarRes)
    addvariable!(prob_large, 2.0)
    for k in 1:NLLSModel.MIN_WORK_PER_THREAD_NORMAL
        addcost!(prob_large, MTTestScalarRes(k * 0.001, 1))
    end
    ls_large_single = NLLSModel.LinearSystem{NLLSModel.UniVariateLSstatic_x{1}, NLLSModel.UniVariateLSstatic_ls{1, 1}}((1,), ())
    @test NLLSModel.workestimates(prob_large, nothing).normal == 2 * NLLSModel.MIN_WORK_PER_THREAD_NORMAL # m = 1, n = 1 per cost
    shared_large = NLLSModel.sharedlinearsystem(ls_large_single, NLLSModel.workestimates(prob_large, nothing).normal, static(4), NLLSModel.costsplit(prob_large.costs))
    @test shared_large isa NLLSModel.LinearSystemShared && length(shared_large.ls) == 2
    @test NLLSModel.sharedlinearsystem(ls_large_single, NLLSModel.workestimates(prob_large, nothing).normal - 1, static(4), NLLSModel.costsplit(prob_large.costs)) isa NLLSModel.LinearSystem
    f_ls, fdf_ls!, x0_ls, _ = optimvars(deepcopy(prob_large); numthreads=static(1))
    f_lm, fdf_lm!, x0_lm, _ = optimvars(deepcopy(prob_large); numthreads=static(4))
    @test f_ls(x0_ls) ≈ f_lm(x0_lm)
    g_ls = zeros(MVector{1, Float64})
    g_lm = zeros(MVector{1, Float64})
    H_ls = zeros(MMatrix{1, 1, Float64, 1})
    H_lm = zeros(MMatrix{1, 1, Float64, 1})
    @test fdf_ls!(g_ls, H_ls, x0_ls) ≈ fdf_lm!(g_lm, H_lm, x0_lm)
    @test g_ls ≈ g_lm
    @test H_ls ≈ H_lm
end

# ---- W1: three work estimates, one per kind of work, from the unfixed variables' dimensions ----

@testset "W1 work estimates" begin
    @testset "known models" begin
        # Bundle adjustment: per cost 2, 2 * 13, 2 * 12 * 13
        ba = wbaproblem(3, 10)
        @test wtuple(NLLSModel.workestimates(ba, nothing)) == 30 .* (2, 2 * 13, 2 * 12 * 13)
        # Procrustes at d = 26: per cost 26, 26 * 326, 26 * 325 * 326
        pr = wprocrustes(26, 5)
        @test wtuple(NLLSModel.workestimates(pr, nothing)) == 5 .* (26, 26 * 326, 26 * 325 * 326)
        @test wtuple(NLLSModel.workestimates(pr, 1)) == wtuple(NLLSModel.workestimates(pr, nothing))
        # Dynamic variables of lengths 2, 3, 5
        dy = wdynproblem()
        @test wtuple(NLLSModel.workestimates(dy, nothing)) == (4 + 1 + 2, 4 * 6 + 1 * 9 + 2 * 8, 4 * 5 * 6 + 1 * 8 * 9 + 2 * 7 * 8)
        # What setup stores is what the estimates say
        f, fdf!, _, _ = optimvars(ba; numthreads=static(4))
        @test f.work == NLLSModel.workestimates(ba, nothing).cost
        _, res!, resjac!, _, _ = optimvars_nls(ba; numthreads=static(4))
        @test res!.layout.work == resjac!.layout.work == NLLSModel.workestimates(ba, nothing).tall
        _, res!, _, _, _ = optimvars_nls(dy; numthreads=static(4), unfixed=[1, 3])
        @test res!.layout.work == NLLSModel.workestimates(dy, [1, 3]).tall
        # A general cost counts as m = 1 with a Hessian, and the standalone cost estimate needs no variables
        @test NLLSModel.costestimate(ba.costs) == NLLSModel.workestimates(ba, nothing).cost
    end

    @testset "fixing a variable lowers tall and linearization by exactly its columns" begin
        ba = wbaproblem(3, 10)
        all_ = NLLSModel.workestimates(ba, nothing)
        mask = trues(length(ba.variables)); mask[1] = false # camera 1, on 10 costs: n goes from 12 to 3
        fixed = NLLSModel.workestimates(ba, mask)
        @test fixed.cost == all_.cost
        @test all_.tall - fixed.tall == 10 * 2 * 9
        @test all_.normal - fixed.normal == 10 * 2 * (12 * 13 - 3 * 4)
        dy = wdynproblem() # fix the 5-vector, on costs 2 (n 8 -> 3) and 3 (n 7 -> 2)
        a, b = NLLSModel.workestimates(dy, nothing), NLLSModel.workestimates(dy, [1, 2])
        @test a.cost == b.cost
        @test a.tall - b.tall == 1 * 5 + 2 * 5
        @test a.normal - b.normal == 1 * (8 * 9 - 3 * 4) + 2 * (7 * 8 - 2 * 3)
    end

    # The one rule: one thread below twice the floor, work ÷ floor between, numthreads above, every thread at floor 0
    @testset "threadcount" begin
        tc(work, floor, nt) = NLLSModel.threadcount(work, floor, nt, 10^6) # Plenty of costs
        @test tc(199, 100, 8) == 1 && tc(0, 100, 8) == 1
        @test tc(200, 100, 8) == 2 && tc(299, 100, 8) == 2 && tc(300, 100, 8) == 3 && tc(799, 100, 8) == 7
        @test tc(800, 100, 8) == 8 && tc(10^9, 100, 8) == 8
        @test tc(0, 0, 8) == 8 && tc(5, 0, 3) == 3 && tc(5, -1, 3) == 3
        @test tc(10^9, 1, 1) == 1 && tc(10^9, 0, 1) == 1
        # Never more threads than the largest number of costs of one type
        @test NLLSModel.threadcount(10^9, 1, 8, 3) == 3 && NLLSModel.threadcount(10^9, 0, 8, 2) == 2
        @test NLLSModel.threadcount(10^9, 0, 8, 1) == 1 && NLLSModel.threadcount(10^9, 0, 8, 0) == 1
        @test NLLSModel.threadcount(10^9, 1, 2, 5) == 2 # numthreads still bounds it
        @test NLLSModel.normalthreads(10^9, 1, 8, 3) == 3 && NLLSModel.normalthreads(10^9, 1, 8, 10) == 4
        # costsplit: the largest number of costs of one type
        mixed = Model(Float64, Union{MTTestScalarRes, MTTestPairRes})
        foreach(_ -> addvariable!(mixed, 1.0), 1:3)
        addcost!(mixed, MTTestScalarRes(1.0, 1))
        addcost!(mixed, MTTestPairRes(0.5, 1, 2)); addcost!(mixed, MTTestPairRes(0.5, 2, 3))
        @test NLLSModel.costsplit(mixed.costs) == 2
        @test NLLSModel.costsplit(Model(Float64, MTTestScalarRes).costs) == 0
        f, fdf!, x, _ = optimvars(mixed; numthreads=static(8), min_work=0, min_work_cost=0)
        @test f.nthreads == 2 && length(fdf!.linsystem.ls) == 2
        @test optimvars_nls(mixed; numthreads=static(8), min_work=0)[2].nthreads == 2
    end

    # A single costly cost is evaluated on one thread, whatever the floors: the other threads would have nothing to do
    @testset "one cost, one thread" begin
        pr = wprocrustes(10, 1)
        f, fdf!, _, _ = optimvars(pr; numthreads=static(8), min_work=0, min_work_cost=0)
        @test f.nthreads == 1 && fdf!.linsystem isa NLLSModel.LinearSystem
        _, res!, _, _, _ = optimvars_nls(pr; numthreads=static(8), min_work=0)
        @test res!.nthreads == 1
        @test NLLSModel.cost(pr.variables, pr.costs, static(8)) == NLLSModel.cost(pr.variables, pr.costs, static(1))
    end

    # Each kind of work decides on its own estimate, by the same rule: floors set for exactly 2 of 8 threads give 2
    # per-thread copies of the linear system, 2 chunks of the tall fill and a 2-way cost sum. The normal fill uses at most
    # MAX_THREADS_NORMAL threads.
    @testset "each kind of work uses the rule on its own estimate" begin
        ba = wbaproblem(3, 10) # cost 60, tall 780, linearization 9360
        w = NLLSModel.workestimates(ba, nothing)
        @test w.cost < w.tall < w.normal
        counts(Fcost, Flin, Ftall) = begin
            f, fdf!, _, _ = optimvars(ba; numthreads=static(8), min_work=Flin, min_work_cost=Fcost)
            _, res!, resjac!, _, _ = optimvars_nls(ba; numthreads=static(8), min_work=Ftall)
            @test res!.nthreads == resjac!.nthreads
            copies = fdf!.linsystem isa NLLSModel.LinearSystemShared ? length(fdf!.linsystem.ls) : 1
            (f.nthreads, copies, res!.nthreads)
        end
        @test counts(w.cost ÷ 2, w.normal ÷ 2, w.tall ÷ 2) == (2, 2, 2)
        @test NLLSModel.MAX_THREADS_NORMAL == 4
        @test counts(w.cost ÷ 3, w.normal ÷ 3, w.tall ÷ 4) == (3, 3, 4)
        @test counts(w.cost ÷ 5, w.normal ÷ 5, w.tall ÷ 5) == (5, 4, 5) # The normal fill is capped
        @test counts(w.cost, w.normal, w.tall) == (1, 1, 1) # Below twice the floor
        @test counts(1, 1, 1) == (8, 4, 8) && counts(0, 0, 0) == (8, 4, 8)
        @test counts(10^9, 10^9, 10^9) == (1, 1, 1)
        # One floor in each unit gives a different count for each kind of work
        @test counts(30, 30, 30) == (2, 4, 8)
        @test counts(400, 400, 400) == (1, 4, 1)
        @test NLLSModel.normalthreads(10^9, 1, 8, 10^6) == 4 && NLLSModel.normalthreads(10^9, 1, 3, 10^6) == 3 && NLLSModel.normalthreads(299, 100, 8, 10^6) == 2
        # numthreads bounds it
        f, fdf!, _, _ = optimvars(ba; numthreads=static(3), min_work=1, min_work_cost=1)
        @test f.nthreads == 3 && length(fdf!.linsystem.ls) == 3
        # The standalone cost uses the default cost floor
        @test NLLSModel.cost(ba.variables, ba.costs, static(4)) ≈ NLLSModel.cost(ba.variables, ba.costs, static(1)) rtol=1e-13
    end

    # Results agree between one thread and the count the rule chose, on every path (the Schur path is in optimization.jl)
    @testset "one thread and the chosen count agree: $name" for (name, make, unfixed) in (
            ("dense", () -> wbaproblem(3, 10), nothing), ("sparse", () -> wbaproblem(12, 40), nothing),
            ("single variable", wdynproblem, 3), ("dynamic", wdynproblem, nothing))
        p = make()
        w = NLLSModel.workestimates(p, isnothing(unfixed) ? nothing : unfixed)
        evalnormal(nt) = begin
            m = deepcopy(p)
            f, fdf!, x, H = optimvars(m; unfixed, numthreads=static(nt), min_work=w.normal ÷ 2, min_work_cost=max(1, w.cost ÷ 2))
            n = Int(sum(v -> dynamic(NLLSModel.nvars(v)), unfixed === nothing ? m.variables : [m.variables[unfixed]]))
            g = H isa AbstractMatrix ? zero(H[:, 1]) : zeros(n)
            Hb = H === nothing ? zeros(n, n) : deepcopy(H)
            c = fdf!(g, Hb, x)
            (f(x), c, Vector(g), Matrix(Hb)), (fdf!.linsystem isa NLLSModel.LinearSystemShared ? length(fdf!.linsystem.ls) : 1)
        end
        evaltall(nt) = begin
            m = deepcopy(p)
            f, res!, resjac!, x, J = optimvars_nls(m; unfixed, numthreads=static(nt), min_work=w.tall ÷ 2, min_work_cost=max(1, w.cost ÷ 2))
            r = zeros(resjac!.layout.M)
            c = resjac!(r, J, x)
            (f(x), c, copy(r), Matrix(J)), resjac!.nthreads
        end
        (s, _), (t, copies) = evalnormal(1), evalnormal(8)
        @test copies == 2
        @test t[1] ≈ s[1] rtol=1e-13
        @test t[2] ≈ s[2] rtol=1e-13
        @test t[3] ≈ s[3] rtol=1e-12
        @test t[4] ≈ s[4] rtol=1e-12
        (s, _), (t, chunks) = evaltall(1), evaltall(8)
        @test chunks == 2
        @test t[3] == s[3] && t[4] == s[4]
        @test t[1] ≈ s[1] rtol=1e-13
        @test t[2] ≈ s[2] rtol=1e-13
    end

    @testset "Procrustes at d = 26 threads its linearization" begin
        pr = wprocrustes(26, 20)
        # The old estimate (residual length times number of variables squared) was 26 * 20 * 1^2 = 520, below the default
        # floor of 10 000, so this did not thread; the linearization estimate is 26 * 325 * 326 * 20.
        @test 26 * 20 * 1^2 < NLLSModel.MIN_WORK_PER_THREAD_NORMAL
        @test NLLSModel.workestimates(pr, nothing).normal >= NLLSModel.MIN_WORK_PER_THREAD_NORMAL
        _, fdf!, _, _ = optimvars(pr; numthreads=static(4))
        @test fdf!.linsystem isa NLLSModel.LinearSystemShared
    end

    # The same numbers with and without threading, forcing each way with a floor of 0 and a very large floor. The
    # residual and Jacobian rows are written by one task each, so they are equal; sums over tasks are equal to rounding.
    @testset "same results threaded and not: $name" for (name, make) in (("bundle adjustment", () -> wbaproblem(3, 10)), ("Procrustes d = 26", () -> wprocrustes(26, 20)), ("dynamic", wdynproblem))
        evalnormal(nt, floor) = begin
            p = make()
            f, fdf!, x, H = optimvars(p; numthreads=static(nt), min_work=floor, min_work_cost=floor)
            n = Int(sum(v -> dynamic(NLLSModel.nvars(v)), p.variables))
            g = zeros(n); Hb = H === nothing ? zeros(n, n) : H
            c = fdf!(g, Hb, x)
            (f(x), c, copy(g), Matrix(Hb))
        end
        evaltall(nt, floor) = begin
            p = make()
            f, res!, resjac!, x, J = optimvars_nls(p; numthreads=static(nt), min_work=floor)
            r = zeros(resjac!.layout.M)
            c = resjac!(r, J, x)
            (f(x), c, copy(r), Matrix(J))
        end
        serial, forced = evalnormal(1, 10^12), evalnormal(4, 0)
        @test forced[1] ≈ serial[1] rtol=1e-13
        @test forced[2] ≈ serial[2] rtol=1e-13
        @test forced[3] ≈ serial[3] rtol=1e-12
        @test forced[4] ≈ serial[4] rtol=1e-12
        @test evalnormal(4, 10^12) == serial # a very large floor forces one thread: bit-identical
        serial, forced = evaltall(1, 10^12), evaltall(4, 0)
        @test forced[3] == serial[3] && forced[4] == serial[4]
        @test forced[2] ≈ serial[2] rtol=1e-13
        # A very large floor makes the fill serial: bit-identical. The cost function f threads on its own estimate at the
        # default cost floor, whatever min_work is, so its sum may differ in the last bits.
        onethread = evaltall(4, 10^12)
        @test onethread[2:4] == serial[2:4]
        @test onethread[1] ≈ serial[1] rtol=1e-13
    end

    # Each cost is linearized once on several threads, as on one: a bipartite model where every cost spans the two sides
    # (on 2 threads the old block-column plan gave one thread every camera and the other every point, and both linearized
    # every cost)
    @testset "each cost linearized once on $nt threads, $storage" for nt in (2, 4), storage in (:dense, :sparse)
        p = storage === :dense ? wbaproblem(3, 10) : wbaproblem(6, 60)
        evals(numthreads) = begin
            _, fdf!, x, H = optimvars(p; numthreads=static(numthreads), min_work=0)
            n = Int(sum(v -> dynamic(NLLSModel.nvars(v)), p.variables))
            g = zeros(n); Hb = H === nothing ? zeros(n, n) : H
            (H === nothing) == (storage === :dense) || error("expected $storage storage")
            fdf!(g, Hb, x) # warm
            WBA_EVALS[] = 0
            fdf!(g, Hb, x)
            WBA_EVALS[]
        end
        serial = evals(1)
        @test serial > 0
        @test evals(nt) == serial
    end
end
