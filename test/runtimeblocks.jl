using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra

# Chain residual between neighbouring 2-vectors, with static (SVector) or run-time (Vector) sized variables
struct ChainRes{V} <: NLLSModel.AbstractResidual
    meas::SVector{2, Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::ChainRes) = static(2)
NLLSModel.nres(::ChainRes) = static(2)
NLLSModel.varindices(r::ChainRes) = SVector(r.i, r.j)
NLLSModel.getvars(r::ChainRes{V}, vars::Vector) where V = (vars[r.i]::V, vars[r.j]::V)
NLLSModel.computeresidual(r::ChainRes, a, b) = SVector(a[1] - b[1] + 0.1 * a[2] * b[2], a[2] - b[2]) - r.meas
Base.eltype(::ChainRes) = Float64

function chainproblem(::Type{V}, n) where V
    problem = Model(V, ChainRes{V})
    for i in 1:n
        addvariable!(problem, V([sin(i), cos(2i)]))
    end
    for i in 1:n-1
        addcost!(problem, ChainRes{V}(SVector(0.1i, -0.2), i, i + 1))
    end
    return problem
end

@testset "runtimeblocks.jl" begin
    for n in (6, 60) # dense and sparse linear systems
        ps = chainproblem(SVector{2, Float64}, n)
        pd = chainproblem(Vector{Float64}, n)
        fs, fdfs!, xs, Hs = optimvars(ps, nothing, static(1))
        fd, fdfd!, xd, Hd = optimvars(pd, nothing, static(1))
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
        @test (@allocated fdfs!(gs, Hs_, xs)) == 0
    end

    # Assembly of a single cost's blocks
    for V in (SVector{2, Float64}, Vector{Float64})
        problem = chainproblem(V, 6)
        _, fdf!, x, _ = optimvars(problem, nothing, static(1))
        ls = fdf!.linsystem
        A = NLLSModel.getA(ls)
        b_vec = NLLSModel.getb(ls)
        blockindices = SVector{2, UInt}(1, 2)
        vars = (problem.variables[1], problem.variables[2])
        a = rand(4, 4)
        b = rand(4)
        boffsets = ls.x.boffsets
        NLLSModel.updatesymA!(A, a, vars, static(3), blockindices)
        NLLSModel.updateb!(b_vec, b, vars, static(3), boffsets, blockindices)
        @test (@allocated NLLSModel.updatesymA!(A, a, vars, static(3), blockindices)) == 0
        @test (@allocated NLLSModel.updateb!(b_vec, b, vars, static(3), boffsets, blockindices)) == 0
    end
end

# Residuals of length one may return a number or a one-element vector (M5)
struct OneRes{S} <: NLLSModel.AbstractResidual
    meas::Float64
end
NLLSModel.ndeps(::OneRes) = static(1)
NLLSModel.nres(::OneRes) = static(1)
NLLSModel.varindices(::OneRes) = 1
NLLSModel.getvars(::OneRes, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(r::OneRes{true}, x) = x[1] * x[2] - r.meas
NLLSModel.computeresidual(r::OneRes{false}, x) = SVector(x[1] * x[2] - r.meas)
Base.eltype(::OneRes) = Float64

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
