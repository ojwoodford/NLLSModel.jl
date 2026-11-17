using NLLSModel, Test, Static, StaticArrays, LinearAlgebra, Random
import SecondOrderOptimization: optimize!, OptimizationOptions

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
            M > NLLSModel.MAX_STATIC_ROWS && push!(variants, (Matrix(J), SVector{M, T}(r))) # Above the row limit: not strided, so plain loops
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
        kernel(J, r) = NLLSModel.gram_kernel(zeros(eltype(J), size(J, 2), size(J, 2)), zeros(eltype(J), size(J, 2)), J, r)
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
        @test allocs_warm(gram, H, g, Jt, r, 0.3, 0.0) == 0
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
            @test allocs_warm(gram, H, g, ∇', r, dc, c2) == 0
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

include("fixtures/gramkernels.jl")

@testset "computerescostgradhess equals the previous implementation" begin
    rng = MersenneTwister(11)
    x = randn(rng, 24)
    meas(M) = randn(rng, M)
    kernels = (NoRobust(), HuberKernel(0.5), Huber2oKernel(0.5), GemanMcclure2oKernel(0.6))
    @testset "$(typeof(kernel)), $name" for kernel in kernels, (name, make) in (
            ("static length 3", () -> GramRes{3, typeof(kernel)}(meas(3), kernel)),
            ("static length 16 (limit)", () -> GramRes{16, typeof(kernel)}(meas(16), kernel)),
            ("static length 20 (above the limit)", () -> GramRes{20, typeof(kernel)}(meas(20), kernel)),
            ("run-time length", () -> GramRes{0, typeof(kernel)}(meas(9), kernel)),
            ("scalar", () -> GramScalar{typeof(kernel)}(0.4, kernel)))
        res = make()
        c, g, H = NLLSModel.computerescostgradhess(static(2), res, kernel, (x,))
        c_ref, g_ref, H_ref = reference_costgradhess(static(2), res, kernel, (x,))
        @test c ≈ c_ref
        @test g ≈ g_ref
        @test H ≈ H_ref
        @test H == H'
    end

    @testset "adaptive kernel" begin
        kernel = ContaminatedGaussian(0.5, 5.0, 0.4)
        for M in (3, 9), varflags in (static(3), static(2)) # With and without the kernel parameters
            res = GramAdaptive{M}(meas(M))
            c, g, H = NLLSModel.computerescostgradhess(varflags, res, kernel, (x,))
            c_ref, g_ref, H_ref = reference_costgradhess(varflags, res, kernel, (x,))
            @test c ≈ c_ref
            @test g ≈ g_ref
            @test H ≈ H_ref
            @test H == H'
        end
    end

end

@testset "no per-cost allocation after warm-up" begin
    rng = MersenneTwister(5)
    x = randn(rng, 24)
    cg(varflags, res, kernel, vars) = NLLSModel.computerescostgradhess(varflags, res, kernel, vars)
    h(M, kernel) = GramHand{M}(randn(rng, M == 0 ? 9 : M), 24, kernel)
    cases = (("no kernel, static length", h(3, NoRobust()), static(2), NoRobust()),
             ("robust kernel, static length", h(9, Huber2oKernel(0.5)), static(2), Huber2oKernel(0.5)),
             ("robust kernel, run-time length", h(0, GemanMcclure2oKernel(0.6)), static(2), GemanMcclure2oKernel(0.6)),
             ("long static residual", h(20, NoRobust()), static(2), NoRobust()))
    for (name, res, flags, kernel) in cases
        cg(flags, res, kernel, (x,))
        cg(flags, res, kernel, (x,))
        @test allocs_warm(cg, flags, res, kernel, (x,)) == 0
    end
    # Adaptive kernel (hand-written Jacobian)
    ares = GramAdaptiveHand{9}(randn(rng, 9), zeros(9), zeros(9, 24))
    akernel = ContaminatedGaussian(0.5, 5.0, 0.4)
    cg(static(3), ares, akernel, (x,))
    cg(static(3), ares, akernel, (x,))
    @test allocs_warm(cg, static(3), ares, akernel, (x,)) == 0
end

@testset "no BLAS call for a Matrix Jacobian" begin
    # The typed code of the dynamic branch contains no call into BLAS and no mul!
    for (M, kernel, flags) in ((3, NoRobust(), static(2)), (9, Huber2oKernel(0.5), static(2)))
        res = GramRes{M, typeof(kernel)}(zeros(M), kernel)
        jac = zeros(M, 5); r = zeros(M)
        for (f, args) in ((NLLSModel.computerescostgradhess_dynamic, (flags, kernel, NLLSModel.staticresidual(r, res), jac, 1.0)),
                          (NLLSModel.gramgrad!, (zeros(5, 5), zeros(5), jac, NLLSModel.staticresidual(r, res), 1.0, 0.0)))
            for (ci, _) in Base.code_typed(f, typeof.(args); optimize=true)
                txt = sprint(show, ci)
                @test !occursin("BLAS", txt) && !occursin("mul!", txt) && !occursin("gemm", txt) && !occursin("gemv", txt)
            end
        end
    end
end

@testset "product cap and kernel selection" begin
    cap = NLLSModel.MAX_STATIC_JAC
    rows = NLLSModel.MAX_STATIC_ROWS
    path(res, x) = begin
        _, jac = NLLSModel.computeresjac(static(1), res, x)
        jac
    end
    n = 32 # Static columns
    x = SVector{n, Float64}(randn(MersenneTwister(1), n))
    @testset "static variables, M * n around MAX_STATIC_JAC" begin
        Mbelow, Mat, Mabove = cap ÷ n - 1, cap ÷ n, cap ÷ n + 1
        # (Running the static path at the cap compiled a 512-element unrolled Jacobian in about 3 minutes, which is why the cap is 128; the numerics are still only run above it)
        for (M, expectstatic) in ((Mbelow, true), (Mat, true), (Mabove, false))
            res = CapRes{M, n}(SVector{M, Float64}(randn(MersenneTwister(2), M)))
            @test NLLSModel.usestatic(static(1), res, (x,)) == expectstatic
        end
        # Above the cap: the dynamic path gives the same cost, gradient and Hessian as before the change (finite differences)
        res = CapRes{Mabove, n}(SVector{Mabove, Float64}(randn(MersenneTwister(2), Mabove)))
        jac = path(res, x)
        @test jac isa Matrix{Float64}
        c, g, H = NLLSModel.computerescostgradhess(static(2), res, NoRobust(), (x,))
        Jf = fdjacobian(res, x)
        r = collect(NLLSModel.computeresidual(res, x))
        @test c ≈ 0.5 * sum(abs2, r)
        @test collect(g) ≈ Jf' * r rtol=1e-6
        @test Matrix(H) ≈ Jf' * Jf rtol=1e-6
        # Below the cap the full static multiply is still used
        small = CapRes{4, 4}(SVector{4, Float64}(randn(MersenneTwister(2), 4)))
        @test path(small, SVector{4, Float64}(1, 2, 3, 4)) isa SMatrix
        # Above the cap with a short static residual (3 x 64): dynamic autodiff, but the static-column kernel, not @turbo
        wide = CapRes{3, 64}(SVector{3, Float64}(1.0, 2.0, 3.0))
        xw = SVector{64, Float64}(randn(MersenneTwister(5), 64))
        @test !NLLSModel.usestatic(static(1), wide, (xw,))
        rw, jw = NLLSModel.computeresjac(static(1), wide, xw)
        @test NLLSModel.staticresidual(rw, wide) isa SVector{3, Float64}
        @test NLLSModel.staticcols_ok(typeof(jw), typeof(NLLSModel.staticresidual(rw, wide)))
    end
    @testset "dynamic variables, static residual around MAX_STATIC_ROWS" begin
        xv = randn(MersenneTwister(3), 40)
        for M in (rows - 1, rows, rows + 1)
            res = GramRes{M, NoRobust}(randn(MersenneTwister(4), M), NoRobust())
            r = NLLSModel.staticresidual(zeros(M), res)
            @test (r isa SVector) == (M <= rows)
            _, jac = NLLSModel.computeresjac(static(1), res, xv)
            @test jac isa Matrix{Float64}
            @test NLLSModel.staticcols_ok(typeof(jac), typeof(r)) == (M <= rows)
        end
    end
    @testset "a run-time residual length takes the dynamic path whatever the variables" begin
        res = CapResDynamic(randn(MersenneTwister(5), 4))
        x4 = SVector{4, Float64}(randn(MersenneTwister(6), 4))
        @test !NLLSModel.usestatic(static(1), res, (x4,))
        _, jac = NLLSModel.computeresjac(static(1), res, x4)
        @test jac isa Matrix{Float64}
    end
    @testset "static variables above the product cap with a long residual: @turbo kernel" begin
        M = rows + 8
        res = CapRes{M, n}(SVector{M, Float64}(randn(MersenneTwister(7), M)))
        # M * n is above the cap and M above the row limit
        @test M * n > cap
        jac = path(res, x)
        r = NLLSModel.staticresidual(zeros(M), res)
        @test jac isa Matrix{Float64} && r isa Vector{Float64}
        @test NLLSModel.turbo_ok(typeof(zeros(n, n)), typeof(zeros(n)), typeof(jac), typeof(r))
    end
end

@testset "tall path with dynamic variables agrees with the linear system" begin
    rng = MersenneTwister(21)
    for kernel in (NoRobust(), Huber2oKernel(0.5), GemanMcclure2oKernel(0.6)), M in (3, 20, 0)
        problem = Model(Vector{Float64}, GramRes{M, typeof(kernel)})
        addvariable!(problem, randn(rng, 24))
        for _ in 1:4
            addcost!(problem, GramRes{M, typeof(kernel)}(randn(rng, M == 0 ? 9 : M), kernel))
        end
        _, fdf!, x0, _ = optimvars(problem; numthreads=static(1))
        g = zeros(24); H = zeros(24, 24)
        c = fdf!(g, H, x0)
        _, _, resjac!, xt, J0 = optimvars_nls(problem; numthreads=static(1))
        r = zeros(size(J0, 1)); J = copy(J0)
        ct = resjac!(r, J, xt)
        @test ct ≈ c
        @test J' * r ≈ g
        @test J' * J ≈ H
    end
end

@testset "B1: staticresidual over the grid" begin
    srows = NLLSModel.MAX_STATIC_ROWS
    inputs(L) = (("Float64", 2.5), ("Vector", randn(L)), ("SVector", SVector{L, Float64}(randn(L))),
                 ("MVector", MVector{L, Float64}(randn(L))), ("view", view(randn(L + 2), 2:L+1)))
    nresvals = (("Int", 5, 5), ("StaticInt at the limit", static(srows), srows), ("StaticInt below the limit", static(3), 3),
                ("StaticInt above the limit", static(srows + 1), srows + 1))
    @testset "$vname, nres is $nname" for (nname, nresval, L) in nresvals, (vname, val) in inputs(L)
        out = NLLSModel.staticresidual(val, NresOf(nresval))
        if val isa Vector && nresval isa StaticInt && dynamic(nresval) <= srows
            @test out isa SVector{L, Float64}
            @test collect(out) == val
        else
            @test out === val
        end
    end
    @testset "no method ambiguities" begin
        ambiguous = Test.detect_ambiguities(NLLSModel; recursive=false)
        @test isempty(filter(pair -> any(m -> m.name in (:staticresidual, :_staticresidual), pair), ambiguous))
    end
end

@testset "B1: (static residual, Matrix Jacobian) end to end" begin
    rng = MersenneTwister(31)
    x = randn(rng, 48)
    kernels = (NoRobust(), HuberKernel(0.5), Huber2oKernel(0.5))
    @testset "$(typeof(kernel)), M = $M, $R" for kernel in kernels, M in (1, 7, 16, 17, 40), R in (:svector, :mvector)
        meas = randn(rng, M)
        res = StatHand{M, R}(meas, 48, kernel)
        ref = StatHand{M, :vector}(meas, 48, kernel) # (Vector, Matrix)
        c, g, H = NLLSModel.computerescostgradhess(static(2), res, kernel, (x,))
        g, H = copy(g), copy(H) # Buffers are reused by the next evaluation
        cv, gv, Hv = NLLSModel.computerescostgradhess(static(2), ref, kernel, (x,))
        c_ref, g_ref, H_ref = reference_costgradhess(static(2), res, kernel, (x,))
        @test c ≈ cv && g ≈ gv && H ≈ Hv
        @test c ≈ c_ref && g ≈ g_ref && H ≈ H_ref
        @test H == H'
        # The kernel reached
        r, J = NLLSModel.computeresjac(static(1), res, x)
        expected = R === :svector && M <= NLLSModel.MAX_STATIC_ROWS ? :staticcols : :plain
        @test kernelof(NLLSModel.staticresidual(r, res), J) == expected
        rv, Jv = NLLSModel.computeresjac(static(1), ref, x)
        expectedv = M <= NLLSModel.MAX_STATIC_ROWS ? :staticcols : :turbo
        @test kernelof(NLLSModel.staticresidual(rv, ref), Jv) == expectedv
    end

    @testset "(Vector, Matrix) with a plain Int nres, M = $M" for M in (3, 40)
        for kernel in kernels
            meas = randn(rng, M)
            res = StatHand{M, :vector}(meas, 48, kernel; dynamicnres=true)
            ref = StatHand{M, :vector}(meas, 48, kernel)
            c, g, H = NLLSModel.computerescostgradhess(static(2), res, kernel, (x,))
            g, H = copy(g), copy(H)
            c_ref, g_ref, H_ref = reference_costgradhess(static(2), res, kernel, (x,))
            @test c ≈ c_ref && g ≈ g_ref && H ≈ H_ref
            r, J = NLLSModel.computeresjac(static(1), res, x)
            @test NLLSModel.staticresidual(r, res) === r # Not converted
            @test kernelof(r, J) == :turbo
        end
    end

    @testset "adaptive kernel, M = $M, $R" for M in (7, 17), R in (:svector, :mvector)
        kernel = ContaminatedGaussian(0.5, 5.0, 0.4)
        meas = randn(rng, M)
        res = StatAdaptive{M, R}(meas, 48)
        for varflags in (static(3), static(2))
            c, g, H = NLLSModel.computerescostgradhess(varflags, res, kernel, (x,))
            g, H = copy(g), copy(H)
            c_ref, g_ref, H_ref = reference_costgradhess(varflags, res, kernel, (x,))
            @test c ≈ c_ref && g ≈ g_ref && H ≈ H_ref
            @test H ≈ H' # (the kernel's own second-derivative block is symmetric only to rounding)
        end
    end

    @testset "against the fully static version, M = $M" for M in (1, 7, 16, 17), kernel in kernels
        N = 6
        xs = SVector{N, Float64}(randn(rng, N))
        meas = SVector{M, Float64}(randn(rng, M))
        stat = StatFull{M, N, true, typeof(kernel)}(meas, kernel)
        dyn = StatFull{M, N, false, typeof(kernel)}(meas, kernel)
        _, J = NLLSModel.computeresjac(static(1), stat, xs)
        @test J isa SMatrix
        cs, gs, Hs = NLLSModel.computerescostgradhess(static(2), stat, kernel, (xs,))
        cd, gd, Hd = NLLSModel.computerescostgradhess(static(2), dyn, kernel, (xs,))
        @test cs ≈ cd && collect(gs) ≈ gd && Matrix(Hs) ≈ Hd
        @test kernelof(meas, zeros(M, N)) == (M <= NLLSModel.MAX_STATIC_ROWS ? :staticcols : :plain)
    end
end

@testset "B1: no allocation for (SVector, Matrix) after warm-up" begin
    x = randn(MersenneTwister(41), 24)
    cg(res, vars) = NLLSModel.computerescostgradhess(static(2), res, NoRobust(), vars)
    for M in (7, 16)
        res = StatHand{M, :svector}(randn(MersenneTwister(42), M), 24, NoRobust())
        cg(res, (x,))
        cg(res, (x,))
        @test allocs_warm(cg, res, (x,)) == 0
    end
end

@testset "B1: a whole solve with (SVector, Matrix) costs, $nthreads thread(s)" for nthreads in (1, 4)
    options = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=50)
    hand, auto = procproblem(true, 300, 51), procproblem(false, 300, 51)
    optimize!(hand, options; numthreads=static(nthreads), min_work=0, min_work_cost=0)
    optimize!(auto, options; numthreads=static(nthreads), min_work=0, min_work_cost=0)
    @test NLLSModel.cost(hand) ≈ NLLSModel.cost(auto) rtol=1e-8
    @test hand.variables[1] ≈ auto.variables[1] atol=1e-6
end

# ---- F2: plain loops for single columns and long static residuals, with @fastmath ----
@testset "F2: kernel reached" begin
    kernel(J, r) = NLLSModel.gram_kernel(zeros(eltype(J), size(J, 2), size(J, 2)), zeros(eltype(J), size(J, 2)), J, r)
    jkernel(J, r) = NLLSModel.jtr_kernel(zeros(eltype(J), size(J, 2)), J, r)
    @testset "n == 1, M = $M" for M in (1, 17, 100, 1000)
        @test kernel(zeros(M, 1), zeros(M)) == :plain
        @test jkernel(zeros(M, 1), zeros(M)) == :plain
        @test kernel(zeros(Float32, M, 1), zeros(Float32, M)) == :plain
    end
    @test kernel(zeros(100, 2), zeros(100)) == :turbo
    @test jkernel(zeros(100, 2), zeros(100)) == :turbo
    @test kernel(zeros(1, 2), zeros(1)) == :turbo
    # The static-column kernel still comes first for a short static residual
    @test kernel(zeros(7, 1), zeros(SVector{7, Float64})) == :staticcols
    @test jkernel(zeros(7, 1), zeros(SVector{7, Float64})) == :staticcols
    # Long static residuals and MVectors with a Matrix stay on the plain kernel, by design
    for r in (zeros(SVector{17, Float64}), zeros(SVector{40, Float64}), zeros(MVector{7, Float64}), zeros(MVector{40, Float64}))
        @test kernel(zeros(length(r), 5), r) == :plain
        @test jkernel(zeros(length(r), 5), r) == :plain
    end
    # An MMatrix Jacobian is not admitted to @turbo either
    @test kernel(zeros(MMatrix{20, 5, Float64}), zeros(20)) == :plain
end

@testset "F2: plain, @turbo and the definition agree, $T" for T in (Float64, Float32)
    rng = MersenneTwister(61)
    tol = T === Float64 ? 1e-12 : 2e-4
    for M in (1, 2, 3, 8, 17, 23, 40, 100), n in (1, 2, 6, 10, 30), (dc, c2) in ((1.0, 0.0), (0.3, -0.05))
        J = randn(rng, T, M, n); r = randn(rng, T, M)
        gu = Float64.(J)' * Float64.(r)
        g_ref = dc * gu
        H_ref = dc * (Float64.(J)' * Float64.(J)) + c2 * gu * gu'
        Hp, gp = zeros(T, n, n), zeros(T, n)
        Ht, gt = zeros(T, n, n), zeros(T, n)
        NLLSModel.gram_plain!(Hp, gp, J, r, T(dc), T(c2))
        NLLSModel.gram_turbo!(Ht, gt, J, r, T(dc), T(c2))
        @test isapprox(Float64.(Hp), H_ref; rtol=tol, atol=tol * max(1, maximum(abs, H_ref)))
        @test isapprox(Float64.(gp), g_ref; rtol=tol, atol=tol * max(1, maximum(abs, g_ref)))
        @test isapprox(Hp, Ht; rtol=10tol, atol=10tol * max(1, maximum(abs, Ht)))
        @test Hp == Hp' # Still exactly symmetric under @fastmath
        jp, jt = zeros(T, n), zeros(T, n)
        NLLSModel.jtr_plain!(jp, J, r); NLLSModel.jtr_turbo!(jt, J, r)
        @test isapprox(Float64.(jp), gu; rtol=tol, atol=tol * max(1, maximum(abs, gu)))
        @test isapprox(jp, jt; rtol=10tol, atol=10tol * max(1, maximum(abs, jt)))
    end
end

@testset "F2: a NaN or Inf reaches g and H through the plain kernels, $T, $bad" for T in (Float64, Float32), bad in (NaN, Inf, -Inf)
    rng = MersenneTwister(62)
    M, n, c = 17, 6, 3
    dc, c2 = T(0.3), T(-0.05)
    # Each runs a plain kernel: called directly, through gramgrad! on an SVector{17} residual and on an Adjoint Jacobian
    runs = (("gram_plain!", (H, g, J, r) -> NLLSModel.gram_plain!(H, g, J, r, dc, c2)),
            ("gramgrad!, SVector{17}", (H, g, J, r) -> NLLSModel.gramgrad!(H, g, J, SVector{17, T}(r), dc, c2)),
            ("gramgrad!, MVector{17}", (H, g, J, r) -> NLLSModel.gramgrad!(H, g, J, MVector{17, T}(r), dc, c2)),
            ("gramgrad!, Adjoint", (H, g, J, r) -> NLLSModel.gramgrad!(H, g, collect(J')', r, dc, c2)))
    @testset "$name, bad entry in J" for (name, run) in runs
        J = randn(rng, T, M, n); r = randn(rng, T, M)
        J[5, c] = bad
        H, g = zeros(T, n, n), zeros(T, n)
        run(H, g, J, r)
        @test !isfinite(g[c])
        @test all(!isfinite, H[c, :]) && all(!isfinite, H[:, c])
        others = setdiff(1:n, c)
        @test all(isfinite, g[others]) && all(isfinite, H[others, others])
    end
    @testset "$name, bad entry in r" for (name, run) in runs
        J = randn(rng, T, M, n); r = randn(rng, T, M)
        r[5] = bad
        H, g = zeros(T, n, n), zeros(T, n)
        run(H, g, J, r)
        @test all(!isfinite, g) && all(!isfinite, H)
    end
    @testset "jtr_plain!" begin
        J = randn(rng, T, M, n); r = randn(rng, T, M)
        J[5, c] = bad
        g = zeros(T, n)
        NLLSModel.jtr_plain!(g, J, r)
        @test !isfinite(g[c]) && all(isfinite, g[setdiff(1:n, c)])
        J[5, c] = 1; r[5] = bad
        NLLSModel.jtr_plain!(g, J, r)
        @test all(!isfinite, g)
    end
    @testset "a single column (routed to plain)" begin
        J = randn(rng, T, 100, 1); r = randn(rng, T, 100)
        J[40, 1] = bad
        H, g = zeros(T, 1, 1), zeros(T, 1)
        NLLSModel.gramgrad!(H, g, J, r, dc, c2)
        @test !isfinite(g[1]) && !isfinite(H[1, 1])
    end
end

@testset "F2: no allocation for the routed cases" begin
    rng = MersenneTwister(63)
    cases = (("n == 1, 100 rows", randn(rng, 100, 1), randn(rng, 100)),
             ("n == 1, 1000 rows", randn(rng, 1000, 1), randn(rng, 1000)),
             ("SVector{17}", randn(rng, 17, 9), SVector{17, Float64}(randn(rng, 17))),
             ("SVector{40}", randn(rng, 40, 9), SVector{40, Float64}(randn(rng, 40))),
             ("MVector{7}", randn(rng, 7, 9), MVector{7, Float64}(randn(rng, 7))),
             ("Float32, n == 1", randn(rng, Float32, 100, 1), randn(rng, Float32, 100)))
    for (name, J, r) in cases
        T = eltype(J); n = size(J, 2)
        H, g = zeros(T, n, n), zeros(T, n)
        f(H, g, J, r) = (NLLSModel.gramgrad!(H, g, J, r, 0.3, -0.05); NLLSModel.jtr!(g, J, r))
        f(H, g, J, r); f(H, g, J, r)
        @test (@allocated f(H, g, J, r)) == 0
    end
end
