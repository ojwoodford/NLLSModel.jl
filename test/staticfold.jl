using NLLSModel, StaticArrays, Static, LinearAlgebra, Test
import OptimizationVariableInterface as OVI
include("fixtures/loader.jl")
fixture("allocations", "staticfold")

# The static single-variable system accumulated in values (src/staticfold.jl): against the in-place accumulation into a
# dynamic system of the same costs (the cost summed in the same order), the kinds of per-cost Hessian,
# inference, allocation.
@testset "staticfold.jl" begin
    @testset "$name" for (name, build, T, noalloc) in STATICFOLD_CASES
        model, unfixed = build()
        vars = model.variables
        f, fdf!, x0, H = optimvars(model; unfixed, numthreads=static(1), precision=T)
        ls = fdf!.linsystem
        @test ls isa NLLSModel.UniVariateLSstatic
        N = size(H, 1)
        g = zeros(MVector{N, Float64})
        c = fdf!(g, H, x0)
        # The in-place path into a dynamic system of the same costs: the cost summed in the same order (bit for bit on
        # Julia 1.13 here; within a few ulp on 1.10, where inlining fuses a residual's muladds differently), g and H to
        # rounding
        cr, gr, Hr = inplace_reference(model, unfixed, N, T)
        @test isapprox(c, cr; rtol=4 * eps())
        @test isapprox(g, gr; rtol=1e-12)
        @test isapprox(H, Hr; rtol=1e-12)
        @test eltype(H) === T && eltype(g) === Float64
        @test H == H' # Both triangles written
        # Inferred, and no allocation
        @test (@inferred NLLSModel.staticfold!(g, H, ls, vars, model.costs)) == c
        noalloc && @test allocs_warm(fdf!, g, H, x0) == 0
        # A Hessian not of the system's type: written from the values, the system's own storage untouched
        internal = NLLSModel.getA(ls)
        fill!(internal, 7)
        gf, Hf = zeros(N), zeros(T, N, N)
        @test fdf!(gf, Hf, x0) == c
        @test gf == g && Hf == H
        @test all(==(7), internal)
    end

    @testset "kinds of per-cost Hessian" begin
        # Each kind the static system can be handed reaches its own method of addhess (or the generic one)
        kinds = Dict(name => kindsof(build()...) for (name, build) in STATICFOLD_CASES)
        @test any(k -> any(t -> t <: NLLSModel.GramLazy{<:Any, <:Adjoint, <:Any, Nothing}, k), values(kinds))
        @test any(k -> any(t -> t <: NLLSModel.GramLazy{<:Any, <:Adjoint, <:Any, Float64}, k), values(kinds))
        @test any(k -> any(t -> t <: NLLSModel.GramLazy{<:Any, <:StaticMatrix, <:Any, Nothing}, k), values(kinds))
        @test any(k -> any(t -> t <: NLLSModel.GramLazy{<:Any, <:StaticMatrix, <:Any, Float64}, k), values(kinds))
        @test any(k -> any(t -> t <: SMatrix, k), values(kinds))
        @test any(k -> any(t -> t <: Matrix, k), values(kinds))
    end

    @testset "summation order" begin
        # More than one block of 1024 costs (the pairwise branch), and the cost summed as `sum` sums it
        model, unfixed = scalarmodel(3, 2500)
        f, fdf!, x0, H = optimvars(model; unfixed, numthreads=static(1))
        g = zero(H[:, 1])
        c = fdf!(g, H, x0)
        cr, gr, Hr = inplace_reference(model, unfixed, 3, Float64)
        @test isapprox(c, cr; rtol=4 * eps()) && isapprox(H, Hr; rtol=1e-12)
        # A single cost (a vector of one)
        model, unfixed = scalarmodel(3, 1)
        f, fdf!, x0, H = optimvars(model; unfixed, numthreads=static(1))
        g = zero(H[:, 1])
        c = fdf!(g, H, x0)
        cr, gr, Hr = inplace_reference(model, unfixed, 3, Float64)
        @test c == cr && g ≈ gr && H ≈ Hr
    end

    @testset "a cost with no unfixed variable" begin
        model, unfixed = nounfixedmodel()
        _, fdf!, x0, H = optimvars(model; unfixed, numthreads=static(1))
        g = zero(H[:, 1])
        c = @test_logs (:warn, r"all variables are fixed") match_mode=:any fdf!(g, H, x0)
        cr, gr, Hr = inplace_reference(model, unfixed, length(g), Float64)
        @test c == cr && g ≈ gr && H ≈ Hr
    end
end
