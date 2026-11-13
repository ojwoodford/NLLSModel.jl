using NLLSModel, Test, Static, StaticArrays, LinearAlgebra, Random

# A1: the native gradient and Gauss-Newton Hessian kernels
@testset "gramkernels.jl" begin
    gram = NLLSModel.gramgrad!
    reference(J, r, dc, c2) = (dc * (J' * r), dc * (J' * J) + c2 * (J' * r) * (J' * r)')
    rng = MersenneTwister(7)

    @testset "agreement with the definition, $T" for T in (Float64, Float32)
        for M in (1, 2, 3, 8, 16, 17, 40), n in (1, 2, 7, 40), (dc, c2) in ((1, 0), (0.3, 0), (0.3, -0.05))
            J = randn(rng, T, M, n)
            r = randn(rng, T, M)
            g_ref, H_ref = reference(Float64.(J), Float64.(r), dc, c2)
            tol = T === Float64 ? 1e-12 : 1e-4
            # Every kernel: static columns (SVector r), @turbo (Vector r), plain loops (non-strided J)
            variants = Any[(Matrix(J), copy(r))]
            M <= NLLSModel.MAX_STATIC_ROWS && push!(variants, (Matrix(J), SVector{M, T}(r)))
            push!(variants, (Matrix(J), SVector{M, T}(r))) # Above the row limit: not strided, so plain loops
            push!(variants, (collect(J')', r)) # An Adjoint (plain loops)
            for (Jv, rv) in variants
                H = fill(T(NaN), n, n)
                g = fill(T(NaN), n)
                gram(H, g, Jv, rv, dc, c2)
                @test H == H' # Exactly symmetric and fully overwritten
                @test all(isfinite, H) && all(isfinite, g)
                @test isapprox(Float64.(H), H_ref; rtol=tol, atol=tol * max(1, maximum(abs, H_ref)))
                @test isapprox(Float64.(g), g_ref; rtol=tol, atol=tol * max(1, maximum(abs, g_ref)))
            end
        end
    end

    @testset "kernel selection" begin
        kernel(J, r) = NLLSModel.staticcols_ok(typeof(J), typeof(r)) ? :staticcols :
                       NLLSModel.turbo_ok(typeof(zeros(2, 2)), typeof(zeros(2)), typeof(J), typeof(r)) ? :turbo : :plain
        J(M, n) = zeros(M, n)
        @test kernel(J(16, 5), zeros(SVector{16, Float64})) == :staticcols
        @test kernel(J(NLLSModel.MAX_STATIC_ROWS, 5), zeros(SVector{NLLSModel.MAX_STATIC_ROWS, Float64})) == :staticcols
        @test kernel(J(17, 5), zeros(SVector{17, Float64})) == :plain # An SVector is not strided; the call site passes a Vector above the limit
        @test kernel(J(17, 5), zeros(17)) == :turbo
        @test kernel(J(3, 5), zeros(3)) == :turbo
        @test kernel(J(3, 5)', zeros(3)) == :plain
        @test kernel(view(J(6, 5), 1:3, :), zeros(3)) == :turbo
        @test kernel(view(J(6, 5), 1:2:5, :), zeros(3)) == :plain # Not unit-stride
        @test kernel(zeros(Float16, 3, 5), zeros(Float16, 3)) == :plain
        @test kernel(zeros(Float32, 3, 5), zeros(Float64, 3)) == :plain # Mixed element types
    end

    @testset "inference, allocation and sizes" begin
        J = randn(rng, 12, 9); r = randn(rng, 12); H = zeros(9, 9); g = zeros(9)
        rs = SVector{12, Float64}(r)
        for rv in (r, rs)
            @inferred gram(H, g, J, rv, 0.3, -0.05)
            gram(H, g, J, rv, 0.3, -0.05)
            @test (@allocated gram(H, g, J, rv, 0.3, -0.05)) == 0
        end
        Jt = collect(J')'
        gram(H, g, Jt, r, 0.3, 0.0)
        @test (@allocated gram(H, g, Jt, r, 0.3, 0.0)) == 0
        @test_throws DimensionMismatch gram(H, g, J, zeros(11), 1.0, 0.0)
        @test_throws DimensionMismatch gram(H, zeros(8), J, r, 1.0, 0.0)
        @test_throws DimensionMismatch gram(zeros(9, 8), g, J, r, 1.0, 0.0)
        @test_throws DimensionMismatch gram(H, g, J, rs[1:11], 1.0, 0.0)
        # Views of a larger buffer (as used for adaptive kernels) are written in place
        Hb = fill(NaN, 11, 11); gb = fill(NaN, 11)
        gram(view(Hb, 3:11, 3:11), view(gb, 3:11), J, r, 0.3, 0.0)
        gram(H, g, J, r, 0.3, 0.0)
        @test Hb[3:11, 3:11] == H && gb[3:11] == g
        @test all(isnan, Hb[1:2, :]) && all(isnan, gb[1:2])
    end

    @testset "scalar residual" begin
        ∇ = randn(rng, 7); r = 0.7
        for (dc, c2) in ((1.0, 0.0), (0.3, -0.05))
            H = fill(NaN, 7, 7); g = fill(NaN, 7)
            gram(H, g, ∇', r, dc, c2)
            g_ref, H_ref = reference(reshape(∇, 1, 7), [r], dc, c2)
            @test H ≈ H_ref && g ≈ g_ref && H == H'
        end
    end

    @testset "static multiply (kernel 1)" begin
        J = randn(rng, SMatrix{5, 3, Float64}); r = randn(rng, SVector{5, Float64})
        for (dc, c2) in ((1.0, 0.0), (0.3, 0.0), (0.3, -0.05))
            g, H = NLLSModel.gramgrad(J, r, dc, c2)
            g_ref, H_ref = reference(J, r, dc, c2)
            @test g ≈ g_ref && H ≈ H_ref
            @test g isa SVector && H isa SMatrix
        end
    end
end
