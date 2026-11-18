using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra

include("fixtures/loader.jl")
fixture("allocations", "runtimeblocks")

# The testsets of runtimeblocks.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
# Threaded dense and sparse linearization (block-column ownership, no per-thread copies) matches the serial result
@testset "threaded block systems" begin
    for (n, copies, V) in ((6, 300, SVector{2, Float64}), (600, 3, SVector{2, Float64}), (6, 300, Vector{Float64}))
        build() = begin
            p = chainproblem(V, n)
            for _ in 2:copies, i in 1:n-1
                addcost!(p, ChainRes{V}(SVector(0.1i, -0.2), i, i + 1))
            end
            addcost!(p, ChainRes{V}(SVector(0.3, 0.1), 1, n)) # A cost spanning the first and last blocks (different threads)
            p
        end
        p1, p4 = build(), build()
        @test NLLSModel.workestimates(p1, nothing).hessianfill >= NLLSModel.MIN_WORK_PER_THREAD_HESSIAN
        f1, fdf1!, x1, H1 = optimvars(p1; numthreads=static(1))
        f4, fdf4!, x4, H4 = optimvars(p4; numthreads=static(4))
        @test fdf4!.linsystem isa NLLSModel.LinearSystemShared # One copy of the system per thread
        g1 = zeros(2n); g4 = zeros(2n)
        H1_ = H1 === nothing ? zeros(2n, 2n) : H1
        H4_ = H4 === nothing ? zeros(2n, 2n) : H4
        for _ in 1:2 # The buffers are re-zeroed on every evaluation
            @test fdf1!(g1, H1_, x1) ≈ fdf4!(g4, H4_, x4)
            @test g1 ≈ g4
            @test Matrix(H1_) ≈ Matrix(H4_)
        end
    end
end

@testset "min_work option" begin
    p = chainproblem(SVector{2, Float64}, 6)
    for _ in 2:300, i in 1:5
        addcost!(p, ChainRes{SVector{2, Float64}}(SVector(0.1i, -0.2), i, i + 1))
    end
    @test optimvars(p; numthreads=static(4))[2].linsystem isa NLLSModel.LinearSystemShared
    @test optimvars(p; numthreads=static(4), min_work=10^9)[2].linsystem isa NLLSModel.LinearSystem
    @test optimvars(p; numthreads=static(4), min_work=1)[2].linsystem isa NLLSModel.LinearSystemShared
end

@testset "work-based thresholds for cost evaluation and the Jacobian path" begin
    p = chainproblem(SVector{2, Float64}, 6)
    for _ in 2:300, i in 1:5
        addcost!(p, ChainRes{SVector{2, Float64}}(SVector(0.1i, -0.2), i, i + 1))
    end
    # Cost evaluation: the floor is an option and the result does not depend on it
    f_default, _, x, _ = optimvars(p; numthreads=static(4))
    f_always, _, _, _ = optimvars(p; numthreads=static(4), min_work_cost=1)
    f_never, _, _, _ = optimvars(p; numthreads=static(4), min_work_cost=10^9)
    @test f_always.min_work == 1 && f_never.min_work == 10^9
    @test f_default(x) ≈ f_always(x) ≈ f_never(x)
    # Jacobian path: work (not cost count) decides whether the evaluators thread, and the layout records it
    _, res1, _, x1, _ = optimvars_nls(p; numthreads=static(1))
    _, res4, _, x4, _ = optimvars_nls(p; numthreads=static(4), min_work=10)
    @test res4.layout.work == NLLSModel.workestimates(p, nothing).jacobianfill
    r1 = zeros(res1.layout.M); r4 = zeros(res4.layout.M)
    @test res1(r1, x1) ≈ res4(r4, x4)
    @test r1 ≈ r4
end

@testset "the threaded dense fdf! fills whichever Hessian matrix it is handed (AL1)" begin
    p = chainproblem(SVector{2, Float64}, 6)
    for _ in 2:300, i in 1:5
        addcost!(p, ChainRes{SVector{2, Float64}}(SVector(0.1i, -0.2), i, i + 1))
    end
    _, fdf!, x, H = optimvars(p; numthreads=static(4))
    @test H === nothing && fdf!.linsystem isa NLLSModel.LinearSystemShared # dense, one copy per thread
    @test alternating_dense_ok(fdf!, x)
end
