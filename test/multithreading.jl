using Test, Static, StaticArrays, SparseArrays, LinearAlgebra, NLLSModel

myfun(x) = x * x * x
NLLSModel.computecost(::StaticInt, ::AbstractVector, x::Float64) = myfun(x)
NLLSModel.computecost(::AbstractVector, x::Float64) = myfun(x)

struct MTTestScalarRes <: NLLSModel.AbstractResidual
    val::Float64
    ind::Int
end
NLLSModel.ndeps(::MTTestScalarRes) = static(1)
NLLSModel.nres(::MTTestScalarRes) = 1
NLLSModel.varindices(res::MTTestScalarRes) = res.ind
NLLSModel.getvars(res::MTTestScalarRes, vars::AbstractVector) = (vars[res.ind]::Float64,)
NLLSModel.computeresidual(res::MTTestScalarRes, x) = x - res.val
Base.eltype(::MTTestScalarRes) = Float64

struct MTTestPairRes <: NLLSModel.AbstractResidual
    val::Float64
    i::Int
    j::Int
end
NLLSModel.ndeps(::MTTestPairRes) = static(2)
NLLSModel.nres(::MTTestPairRes) = static(1)
NLLSModel.varindices(res::MTTestPairRes) = SVector(res.i, res.j)
NLLSModel.getvars(res::MTTestPairRes, vars::AbstractVector) = (vars[res.i]::Float64, vars[res.j]::Float64)
NLLSModel.computeresidual(res::MTTestPairRes, x, y) = SVector(x - y - res.val)
Base.eltype(::MTTestPairRes) = Float64

struct MTTestDynamicRes <: NLLSModel.AbstractResidual
    y::Float64
    X::Vector{Float64}
end
NLLSModel.ndeps(::MTTestDynamicRes) = static(1)
NLLSModel.nres(::MTTestDynamicRes) = static(1)
NLLSModel.varindices(::MTTestDynamicRes) = 1
NLLSModel.getvars(::MTTestDynamicRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(res::MTTestDynamicRes, w) = dot(res.X, w) - res.y
Base.eltype(::MTTestDynamicRes) = Float64

alloc_ephemeral(ls, ind) = @allocated NLLSModel.LinearSystem(ls, ind)

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
        f_s, fdf_s!, x0_s, H_s = optimvars(deepcopy(prob), nothing, static(1))
        f_m, fdf_m!, x0_m, H_m = optimvars(deepcopy(prob), nothing, static(4))
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

    # Threshold-based decision test (< 1e4 stays single-threaded, >= 1e4 creates LinearSystemShared)
    @test isa(NLLSModel.sharedlinearsystem(ls_static_single, prob_static, static(4)), NLLSModel.LinearSystem)
    prob_large = Model(Float64, MTTestScalarRes)
    addvariable!(prob_large, 2.0)
    for k in 1:10000
        addcost!(prob_large, MTTestScalarRes(k * 0.001, 1))
    end
    ls_large_single = NLLSModel.LinearSystem{NLLSModel.UniVariateLSstatic_x{1}, NLLSModel.UniVariateLSstatic_ls{1, 1}}((1,), ())
    @test isa(NLLSModel.sharedlinearsystem(ls_large_single, prob_large, static(4)), NLLSModel.LinearSystemShared)
    f_ls, fdf_ls!, x0_ls, _ = optimvars(deepcopy(prob_large), nothing, static(1))
    f_lm, fdf_lm!, x0_lm, _ = optimvars(deepcopy(prob_large), nothing, static(4))
    @test f_ls(x0_ls) ≈ f_lm(x0_lm)
    g_ls = zeros(MVector{1, Float64})
    g_lm = zeros(MVector{1, Float64})
    H_ls = zeros(MMatrix{1, 1, Float64, 1})
    H_lm = zeros(MMatrix{1, 1, Float64, 1})
    @test fdf_ls!(g_ls, H_ls, x0_ls) ≈ fdf_lm!(g_lm, H_lm, x0_lm)
    @test g_ls ≈ g_lm
    @test H_ls ≈ H_lm
end
