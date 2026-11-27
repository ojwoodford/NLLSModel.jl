using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra

include("fixtures/loader.jl")
fixture("allocations", "runtimeblocks")

@testset "runtimeblocks.jl" begin
    for n in (6, 60) # dense and sparse linear systems
        ps = chainproblem(SVector{2, Float64}, n)
        pd = chainproblem(Vector{Float64}, n)
        fs, fdfs!, xs, Hs = optimvars(ps; numthreads=static(1))
        fd, fdfd!, xd, Hd = optimvars(pd; numthreads=static(1))
        @test (Hs === nothing) == (Hd === nothing)
        @test reduce(vcat, xs) ≈ reduce(vcat, xd)
        @test fs(xs) ≈ fd(xd)
        gs = zeros(2n); gd = zeros(2n)
        Hs_ = Hs === nothing ? zeros(2n, 2n) : Hs
        Hd_ = Hd === nothing ? zeros(2n, 2n) : Hd
        @test fdfs!(gs, Hs_, xs) ≈ fdfd!(gd, Hd_, xd)
        @test gs ≈ gd
        @test Matrix(Hs_) ≈ Matrix(Hd_)
        # Static variables must stay allocation free
        @test allocs_warm(fdfs!, gs, Hs_, xs) == 0
    end

    # Assembly of a single cost's blocks
    for V in (SVector{2, Float64}, Vector{Float64})
        problem = chainproblem(V, 6)
        _, fdf!, x, _ = optimvars(problem; numthreads=static(1))
        ls = fdf!.linsystem
        A = NLLSModel.getA(ls)
        b_vec = NLLSModel.getb(ls)
        blockindices = SVector{2, UInt}(1, 2)
        vars = (problem.variables[1], problem.variables[2])
        a = rand(4, 4)
        b = rand(4)
        boffsets = ls.x.boffsets
        NLLSModel.updatesymA!(A, a, map(NLLSModel.nvars, vars), static(3), blockindices)
        NLLSModel.updateb!(b_vec, b, map(NLLSModel.nvars, vars), static(3), boffsets, blockindices)
        @test allocs_warm(NLLSModel.updatesymA!, A, a, map(NLLSModel.nvars, vars), static(3), blockindices) == 0
        @test allocs_warm(NLLSModel.updateb!, b_vec, b, map(NLLSModel.nvars, vars), static(3), boffsets, blockindices) == 0
    end
end

@testset "dynamic autodiff" begin
    x = [2.0, 3.0]
    for S in (true, false)
        res, jac = NLLSModel.computeresjac(static(1), OneRes{S}(1.0), x)
        @test only(res) ≈ 5.0
        @test vec(Matrix(jac)) ≈ [3.0, 2.0]
    end
    # Repeated calls reuse the cached buffers (allocations are well below a fresh setup)
    cost = ChainRes{Vector{Float64}}(SVector(0.1, -0.2), 1, 2)
    a, b = [1.0, 2.0], [0.5, -1.0]
    NLLSModel.computeresjac(static(3), cost, a, b)
    first_call = @allocated NLLSModel.computeresjac(static(3), cost, a, b)
    @test first_call < 2000
    res, jac = NLLSModel.computeresjac(static(3), cost, a, b)
    res_s, jac_s = NLLSModel.computeresjac(static(3), ChainRes{SVector{2, Float64}}(SVector(0.1, -0.2), 1, 2), SVector{2}(a), SVector{2}(b))
    @test collect(res) ≈ collect(res_s)
    @test Matrix(jac) ≈ Matrix(jac_s)
end


@testset "the dense fdf! fills whichever Hessian matrix it is handed (AL1)" begin
    for V in (SVector{2, Float64}, Vector{Float64})
        _, fdf!, x, H = optimvars(chainproblem(V, 6); numthreads=static(1))
        @test H === nothing && fdf!.linsystem isa NLLSModel.LinearSystem # dense, on one thread
        @test alternating_dense_ok(fdf!, x)
    end
end
