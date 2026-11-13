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
            @test (@allocated gram(H, g, ∇', r, dc, c2)) == 0
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

# ---- computerescostgradhess against the previous implementation (BLAS products and reweighting passes) ----
function reference_costgradhess(varflags, residual, kernel, vars)
    varflagsres = varflags >> static(1)
    res, jac = NLLSModel.computeresjac(varflagsres, residual, vars...)
    sqres = NLLSModel.sqnorm(res)
    g = jac' * res
    H = jac' * jac
    if varflags & 1 == 0
        cost, dc, d2c = robustifydcost(kernel, sqres)
    else
        cost, dc_, d2c_ = robustifydkernel(kernel, sqres)
        dc = dc_[end]
        d2c = d2c_[end, end]
        kernelvarind = NLLSModel.SR(1, NLLSModel.nvars(kernel))
        dkdv = g * view(d2c_, kernelvarind, NLLSModel.nvars(kernel) + 1)'
    end
    dc != 1 && (H *= dc)
    d2c != 0 && (H += (max(2 * d2c, -dc / (3 * sqres)) * g) * g')
    dc != 1 && (g *= dc)
    if varflags & 1 == 1
        g = vcat(0.5 * view(dc_, kernelvarind), g)
        H = hcat(vcat(0.5 * view(d2c_, kernelvarind, kernelvarind), dkdv), vcat(dkdv', H))
    end
    return 0.5 * Float64(cost), g, H
end

# Residual of a variable with run-time size: M entries x[k] * x[k+1] - meas[k], as an SVector (static length) or a Vector
struct GramRes{M, K} <: NLLSModel.AbstractResidual # M == 0: run-time length
    meas::Vector{Float64}
    kernel::K
end
NLLSModel.ndeps(::GramRes) = static(1)
NLLSModel.nres(r::GramRes{M}) where M = M == 0 ? length(r.meas) : static(M)
NLLSModel.varindices(::GramRes) = 1
NLLSModel.getvars(::GramRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.robustkernel(r::GramRes) = r.kernel
Base.eltype(::GramRes) = Float64
NLLSModel.computeresidual(r::GramRes{0}, x) = [x[k] * x[k+1] - r.meas[k] for k in eachindex(r.meas)]
NLLSModel.computeresidual(r::GramRes{M}, x) where M = SVector{M}(ntuple(k -> x[k] * x[k+1] - r.meas[k], Val(M)))

# The same with a scalar residual
struct GramScalar{K} <: NLLSModel.AbstractResidual
    meas::Float64
    kernel::K
end
NLLSModel.ndeps(::GramScalar) = static(1)
NLLSModel.nres(::GramScalar) = static(1)
NLLSModel.varindices(::GramScalar) = 1
NLLSModel.getvars(::GramScalar, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.robustkernel(r::GramScalar) = r.kernel
Base.eltype(::GramScalar) = Float64
NLLSModel.computeresidual(r::GramScalar, x) = sum(abs2, x) - r.meas

# Adaptive-kernel residual of a run-time sized variable
struct GramAdaptive{M} <: NLLSModel.AbstractAdaptiveResidual
    meas::Vector{Float64}
end
NLLSModel.ndeps(::GramAdaptive) = static(2)
NLLSModel.nres(::GramAdaptive{M}) where M = static(M)
NLLSModel.varindices(::GramAdaptive) = SVector(1, 2)
NLLSModel.getvars(::GramAdaptive, vars::AbstractVector) = (vars[1]::ContaminatedGaussian{Float64}, vars[2]::Vector{Float64})
Base.eltype(::GramAdaptive) = Float64
NLLSModel.computeresidual(r::GramAdaptive{M}, x) where M = SVector{M}(ntuple(k -> x[k] * x[k+1] - r.meas[k], Val(M)))

struct GramAdaptiveHand{M} <: NLLSModel.AbstractAdaptiveResidual
    meas::Vector{Float64}
    res::Vector{Float64}
    jac::Matrix{Float64}
end
NLLSModel.ndeps(::GramAdaptiveHand) = static(2)
NLLSModel.nres(::GramAdaptiveHand{M}) where M = static(M)
NLLSModel.varindices(::GramAdaptiveHand) = SVector(1, 2)
NLLSModel.getvars(::GramAdaptiveHand, vars::AbstractVector) = (vars[1]::ContaminatedGaussian{Float64}, vars[2]::Vector{Float64})
Base.eltype(::GramAdaptiveHand) = Float64
function NLLSModel.computeresjac(varflags, r::GramAdaptiveHand{M}, x) where M
    for k in 1:M
        r.res[k] = x[k] * x[k+1] - r.meas[k]
        fill!(view(r.jac, k, :), 0.0)
        r.jac[k, k] = x[k+1]
        r.jac[k, k+1] = x[k]
    end
    return r.res, r.jac
end

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

# Residuals with hand-written Jacobians in buffers they own (as a user with dynamic variables would write them). The
# autodiff path of a `Vector` variable allocates in `update`, which builds a dual vector, so it is not used here.
struct GramHand{M, K} <: NLLSModel.AbstractResidual # M == 0: run-time length
    meas::Vector{Float64}
    res::Vector{Float64}
    jac::Matrix{Float64}
    kernel::K
end
GramHand{M}(meas, n, kernel) where M = GramHand{M, typeof(kernel)}(meas, zeros(length(meas)), zeros(max(length(meas), 1), n), kernel)
NLLSModel.ndeps(::GramHand) = static(1)
NLLSModel.nres(r::GramHand{M}) where M = M == 0 ? length(r.meas) : static(M)
NLLSModel.varindices(::GramHand) = 1
NLLSModel.getvars(::GramHand, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.robustkernel(r::GramHand) = r.kernel
Base.eltype(::GramHand) = Float64
function NLLSModel.computeresjac(varflags, r::GramHand{M}, x) where M
    m = length(r.meas)
    for k in 1:m
        r.res[k] = x[k] * x[k+1] - r.meas[k]
        fill!(view(r.jac, k, :), 0.0)
        r.jac[k, k] = x[k+1]
        r.jac[k, k+1] = x[k]
    end
    return r.res, r.jac
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
        @test (@allocated cg(flags, res, kernel, (x,))) == 0
    end
    # Adaptive kernel (hand-written Jacobian)
    ares = GramAdaptiveHand{9}(randn(rng, 9), zeros(9), zeros(9, 24))
    akernel = ContaminatedGaussian(0.5, 5.0, 0.4)
    cg(static(3), ares, akernel, (x,))
    cg(static(3), ares, akernel, (x,))
    @test (@allocated cg(static(3), ares, akernel, (x,))) == 0
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

# Product cap of the full static path (MAX_STATIC_JAC)
struct CapRes{M, N} <: NLLSModel.AbstractResidual # N static variables of size N, M residuals
    meas::SVector{M, Float64}
end
NLLSModel.ndeps(::CapRes) = static(1)
NLLSModel.nres(::CapRes{M}) where M = static(M)
NLLSModel.varindices(::CapRes) = 1
NLLSModel.getvars(::CapRes{M, N}, vars::AbstractVector) where {M, N} = (vars[1]::SVector{N, Float64},)
Base.eltype(::CapRes) = Float64
NLLSModel.computeresidual(r::CapRes{M, N}, x) where {M, N} = SVector{M}(ntuple(k -> x[mod1(k, N)] * x[mod1(k + 1, N)] - r.meas[k], Val(M)))
# A residual whose length is only known at run time
struct CapResDynamic <: NLLSModel.AbstractResidual
    meas::Vector{Float64}
end
NLLSModel.ndeps(::CapResDynamic) = static(1)
NLLSModel.nres(r::CapResDynamic) = length(r.meas)
NLLSModel.varindices(::CapResDynamic) = 1
NLLSModel.getvars(::CapResDynamic, vars::AbstractVector) = (vars[1]::SVector{4, Float64},)
Base.eltype(::CapResDynamic) = Float64
NLLSModel.computeresidual(r::CapResDynamic, x) = SVector{4}(ntuple(k -> x[mod1(k, 4)] * x[mod1(k + 1, 4)] - r.meas[k], Val(4)))

function fdjacobian(res, x; h=1e-6)
    cols = map(1:length(x)) do i
        e = SVector{length(x), Float64}(ntuple(j -> j == i ? h : 0.0, length(x)))
        collect(NLLSModel.computeresidual(res, x + e) - NLLSModel.computeresidual(res, x - e)) / 2h
    end
    return reduce(hcat, cols)
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
        _, fdf!, x0, _ = optimvars(problem, nothing, static(1))
        g = zeros(24); H = zeros(24, 24)
        c = fdf!(g, H, x0)
        for robust in (kernel isa NoRobust ? (:none, :irls) : (:corrector,))
            _, _, resjac!, xt, J0 = optimvars_nls(problem; robust=robust, numthreads=static(1))
            r = zeros(size(J0, 1)); J = copy(J0)
            ct = resjac!(r, J, xt)
            @test ct ≈ c
            @test J' * r ≈ g
            @test J' * J ≈ H
        end
    end
end
