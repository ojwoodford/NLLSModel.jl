using NLLSModel, Test, Static, StaticArrays, Random
using OptimizationVariableInterface: update

include("fixtures/loader.jl")
fixture("adaptivecost", "kernelconvention")

# The convention: ρ(s) = −2 log p(r), p a density over r ∈ ℝᵈ and s = ‖r‖²; an adaptive kernel returns it exactly
@testset "kernelconvention.jl" begin
    @testset "ContaminatedGaussian is normalized, d = $d, $(params)" for d in 1:3, params in ((0.5, 5.0, 0.4), (1.0, 10.0, 0.8))
        s1, s2, w = params
        @test densityintegral(cgkernel(s1, s2, w, d), d, max(s1, s2)) ≈ 1.0 atol=1e-8
    end

    @testset "ContaminatedGaussian at s = 0 and s = 1e300, d = $d" for d in 1:3
        k = cgkernel(0.5, 5.0, 0.4, d)
        for s in (0.0, 1e300)
            @test isfinite(robustify(k, s))
            @test all(isfinite, robustifydcost(k, s))
        end
    end

    @testset "ContaminatedGaussian's robustifydcost against autodiff, d = $d" for d in 1:3
        k = cgkernel(0.6, 9.0, 0.7, d)
        for s in KERNEL_FD_COSTS
            @test SVector(robustifydcost(k, s)) ≈ SVector(NLLSModel.autorobustifydcost(k, s))
        end
    end

    @testset "addcost! checks the kernel's residual length" begin
        K3 = typeof(cgkernel(0.5, 5.0, 0.4, 3))
        K2 = typeof(cgkernel(0.5, 5.0, 0.4, 2))
        # Adaptive: the kernel is variable 1
        model = Model(Union{K3, SVector{2, Float64}}, PairAdaptive{K3})
        addvariable!(model, cgkernel(0.5, 5.0, 0.4, 3))
        addvariable!(model, SVector(0.0, 0.0))
        err = try addcost!(model, PairAdaptive{K3}(SVector(1.0, 2.0))); nothing catch e e end
        @test err isa ArgumentError && occursin("3", err.msg) && occursin("2", err.msg)
        model = Model(Union{K2, SVector{2, Float64}}, PairAdaptive{K2})
        addvariable!(model, cgkernel(0.5, 5.0, 0.4, 2))
        addvariable!(model, SVector(0.0, 0.0))
        @test addcost!(model, PairAdaptive{K2}(SVector(1.0, 2.0))) === nothing
        # Fixed, through robustkernel, and through Scaled
        for (bad, good) in ((cgkernel(0.5, 5.0, 0.4, 3), cgkernel(0.5, 5.0, 0.4, 2)),
                            (Scaled(cgkernel(0.5, 5.0, 0.4, 3), 2.0), Scaled(cgkernel(0.5, 5.0, 0.4, 2), 2.0)))
            model = Model(SVector{2, Float64}, Union{PairFixed{typeof(bad)}, PairFixed{typeof(good)}})
            addvariable!(model, SVector(0.0, 0.0))
            @test_throws ArgumentError addcost!(model, PairFixed(SVector(1.0, 2.0), bad))
            @test addcost!(model, PairFixed(SVector(1.0, 2.0), good)) === nothing
        end
        # A user's kernel, and the fixed kernels here, have no residual length
        model = Model(SVector{2, Float64}, Union{PairFixed{UserKernel}, PairFixed{HuberKernel{Float64, Static.False}}})
        addvariable!(model, SVector(0.0, 0.0))
        @test addcost!(model, PairFixed(SVector(1.0, 2.0), UserKernel())) === nothing
        @test addcost!(model, PairFixed(SVector(1.0, 2.0), HuberKernel(1.0))) === nothing
    end

    @testset "a residual of length 2: cost, gradient and Hessian against finite differences" begin
        kernel = cgkernel(0.5, 5.0, 0.4, 2)
        res = PairAdaptive{typeof(kernel)}(SVector(0.7, -0.3))
        var = SVector(0.2, 0.1)
        cfd, gfd, Hfd = fdcostgradhess(res, kernel, var)
        c, g, H = NLLSModel.computecostgradhess(static(3), res, kernel, var) # Kernel and variable both optimized
        @test c ≈ cfd
        @test g ≈ gfd rtol=1e-5
        @test H[1:3, 1:3] ≈ Hfd[1:3, 1:3] rtol=1e-3 atol=1e-5
        @test H[1:3, 4:5] ≈ Hfd[1:3, 4:5] rtol=1e-3 atol=1e-5
        c, g, H = NLLSModel.computecostgradhess(static(1), res, kernel, var) # The kernel alone
        @test g ≈ gfd[1:3] rtol=1e-5
        @test H ≈ Hfd[1:3, 1:3] rtol=1e-3 atol=1e-5
        @test c ≈ 0.5 * robustify(kernel, sum(abs2, var - res.data))
    end

    @testset "ContaminatedGaussian evaluates without allocating, d = $d" for d in 1:3
        k = cgkernel(0.6, 9.0, 0.7, d)
        @test allocs_warm(robustify, k, 2.0) == 0
        @test allocs_warm(robustifydcost, k, 2.0) == 0
        @test allocs_warm(robustifydkernel, k, 2.0) == 0
        @test allocs_warm(update, k, SVector(0.1, 0.2, 0.3), 1) == 0
    end

    @testset "GaussianPlusUniform is normalized, d = $d, $(params)" for d in 1:3, params in ((0.5, 0.4, 20.0), (2.0, 0.8, 1e4))
        sigma, w, A = params
        k = gpukernel(sigma, w, d; volume=A)
        @test densityintegral(k, d, sigma; minus=(1 - w) / A) ≈ w atol=1e-8
    end

    @testset "GaussianPlusUniform: derivatives, d = $d" for d in 1:3
        k = gpukernel(0.6, 0.7, d; volume=30.0)
        @testset "s = $s" for s in KERNEL_FD_COSTS
            # In s, against finite differences and autodiff
            c, d1, d2 = robustifydcost(k, s)
            @test c ≈ robustify(k, s)
            @test d1 ≈ fdcost1(k, s) rtol=1e-6
            @test d2 ≈ fdcost2(k, s) rtol=1e-4 atol=1e-30
            @test SVector(c, d1, d2) ≈ SVector(NLLSModel.autorobustifydcost(k, s))
            # In the parameters and s, against finite differences and autodiff, in the autodiff fallback's order and types
            c, g, H = robustifydkernel(k, s)
            gfd, Hfd = fdkernel(k, s)
            @test g ≈ gfd rtol=1e-5
            @test H ≈ Hfd rtol=1e-3 atol=1e-5
            ca, ga, Ha = NLLSModel.autorobustifydkernel(k, s)
            @test c ≈ ca && g ≈ ga && H ≈ Ha
            @test typeof((c, g, H)) == typeof((ca, ga, Ha))
            @test H == H'
        end
    end

    @testset "GaussianPlusUniform: a small model's cost, gradient and Hessian, d = $d" for d in 1:3
        kernel = gpukernel(0.5, 0.6, d; volume=40.0)
        res = VecAdaptive{d, typeof(kernel)}(SVector{d}(ntuple(i -> 0.7 / i - 0.3, d)))
        var = SVector{d}(ntuple(i -> 0.2 / i, d))
        cfd, gfd, Hfd = fdcostgradhess(res, kernel, var)
        c, g, H = NLLSModel.computecostgradhess(static(3), res, kernel, var)
        @test c ≈ cfd
        @test g ≈ gfd rtol=1e-5
        @test H[1:2, 1:2] ≈ Hfd[1:2, 1:2] rtol=1e-3 atol=1e-5
        @test H[1:2, 3:end] ≈ Hfd[1:2, 3:end] rtol=1e-3 atol=1e-5
        c, g, H = NLLSModel.computecostgradhess(static(1), res, kernel, var)
        @test g ≈ gfd[1:2] rtol=1e-5
        @test H ≈ Hfd[1:2, 1:2] rtol=1e-3 atol=1e-5
    end

    @testset "GaussianPlusUniform at s = 0 and s = 1e300, d = $d" for d in 1:3
        k = gpukernel(0.5, 0.4, d)
        for s in (0.0, 1e300)
            @test isfinite(robustify(k, s))
            @test all(isfinite, robustifydcost(k, s))
            c, g, H = robustifydkernel(k, s)
            @test isfinite(c) && all(isfinite, g) && all(isfinite, H)
        end
    end

    @testset "GaussianPlusUniform: construction, withlogvolume, update" begin
        k = gpukernel(0.5, 0.4, 2; volume=50.0)
        @test k isa GaussianPlusUniform{Float64, 2}
        @test NLLSModel.params(k) ≈ SVector(0.5, 0.4)
        @test k.logvolume ≈ log(50.0)
        @test GaussianPlusUniform(1, 0.5f0, static(1); volume=2) isa GaussianPlusUniform{Float32, 1}
        k2 = withlogvolume(k, log(7.0))
        @test typeof(k2) === typeof(k)
        @test k2.logvolume == log(7.0)
        for f in (:invsigma, :w, :halfinvvar, :k, :log1mw)
            @test getfield(k2, f) === getfield(k, f)
        end
        @test withlogvolume(k, log(7.0)) == gpukernel(0.5, 0.4, 2; volume=7.0)
        @test withlogvolume(k, 1) isa GaussianPlusUniform{Float64, 2}
        u = update(k, SVector(0.1, -0.2), 1)
        @test u isa GaussianPlusUniform{Float64, 2} && u.logvolume == k.logvolume
        @test NLLSModel.params(u) ≈ SVector(0.5 * exp(-0.1), 1 / (1 + exp(0.2) * (1 - 0.4) / 0.4))
        @test NLLSModel.kerneldim(k) == 2 && NLLSModel.nvars(k) == static(2)
        # Errors
        @test_throws UndefKeywordError GaussianPlusUniform(0.5, 0.4, static(2))
        for A in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError GaussianPlusUniform(0.5, 0.4, static(2); volume=A)
        end
        @test_throws ArgumentError GaussianPlusUniform(0.5, 0.4, 2; volume=1.0)
        @test_throws ArgumentError GaussianPlusUniform(0.5, 0.4, static(0); volume=1.0)
        @test_throws ArgumentError GaussianPlusUniform(k.invsigma, k.w, 2, 0.0)
        # addcost! checks the residual length
        K3 = typeof(gpukernel(0.5, 0.4, 3))
        model = Model(Union{K3, SVector{2, Float64}}, PairAdaptive{K3})
        addvariable!(model, gpukernel(0.5, 0.4, 3))
        addvariable!(model, SVector(0.0, 0.0))
        @test_throws ArgumentError addcost!(model, PairAdaptive{K3}(SVector(1.0, 2.0)))
        model = Model(SVector{2, Float64}, PairFixed{K3})
        addvariable!(model, SVector(0.0, 0.0))
        @test_throws ArgumentError addcost!(model, PairFixed(SVector(1.0, 2.0), gpukernel(0.5, 0.4, 3)))
    end

    @testset "GaussianPlusUniform: a residual's own volume through robustkernel" begin
        K = gpukernel(0.5, 0.4, 2; volume=50.0)
        x = SVector(0.3, -0.2)
        model = Model(SVector{2, Float64}, PairVolume{typeof(K)})
        addvariable!(model, x)
        data = (SVector(0.1, 0.2), SVector(3.0, -4.0), SVector(0.4, 0.0))
        logvolumes = (log(2.0), log(500.0), log(50.0))
        for (y, l) in zip(data, logvolumes)
            addcost!(model, PairVolume(y, l, K))
        end
        w, s = 0.4, 0.5
        byhand = sum(zip(data, logvolumes)) do (y, l)
            r2 = sum(abs2, x - y)
            p = w * exp(-r2 / (2s^2)) / (2π * s^2) + (1 - w) / exp(l)
            -log(p) # 0.5 ρ
        end
        @test cost(model) ≈ byhand
    end

    @testset "GaussianPlusUniform evaluates without allocating" begin
        k = gpukernel(0.6, 0.7, 2)
        @test allocs_warm(robustify, k, 2.0) == 0
        @test allocs_warm(robustifydcost, k, 2.0) == 0
        @test allocs_warm(robustifydkernel, k, 2.0) == 0
        @test allocs_warm(withlogvolume, k, 1.0) == 0
        @test allocs_warm(update, k, SVector(0.1, 0.2), 1) == 0
        # The Hessian path, the kernel fixed (varflags 2) and unfixed (3): a static Jacobian and a run-time sized one
        rs = PairAdaptive{typeof(k)}(SVector(0.7, -0.3))
        rd = PairAdaptiveDynamic{typeof(k)}(SVector(0.7, -0.3))
        for flags in (static(2), static(3))
            @test allocs_warm(costgradhess, flags, rs, k, SVector(0.2, 0.1)) == 0
            @test allocs_warm(costgradhess, flags, rd, k, [0.2, 0.1, 0.5]) == 0
        end
    end

    @testset "the docstring examples of GaussianPlusUniform and withlogvolume" begin
        kernel = GaussianPlusUniform(1.0, 0.9, static(2); volume = 100.0)
        @test isfinite(robustify(kernel, 0.5))
        @test withlogvolume(kernel, log(250.0)) == GaussianPlusUniform(1.0, 0.9, static(2); volume = 250.0)
    end

    # Each kernel's derivatives in s against central differences of robustify (in BigFloat, so that rounding plays no
    # part: an absolute floor of 1e-30 covers a second derivative that is exactly zero)
    @testset "robustifydcost against finite differences, $name" for (name, kernel, firstorder) in (
            ("NoRobust", NoRobust(), false), ("Scaled", Scaled(Huber2oKernel(1.0), 2.0), false),
            ("HuberKernel", HuberKernel(1.0), true), ("Huber2oKernel", Huber2oKernel(1.0), false),
            ("GemanMcclureKernel", GemanMcclureKernel(0.6), true), ("GemanMcclure2oKernel", GemanMcclure2oKernel(0.6), false),
            ("ContaminatedGaussian", cgkernel(0.6, 9.0, 0.7, 1), false))
        @testset "s = $s" for s in KERNEL_FD_COSTS
            c, d1, d2 = robustifydcost(kernel, s)
            @test c ≈ robustify(kernel, s)
            @test d1 ≈ fdcost1(kernel, s) rtol=1e-6
            if firstorder
                @test d2 === 0.0 # By design, not the difference
            else
                @test d2 ≈ fdcost2(kernel, s) rtol=1e-4 atol=1e-30
            end
        end
    end

    # The kernel's derivatives in its parameters and s against central differences of robustify(update(kernel, x), s + x[end])
    @testset "robustifydkernel against finite differences, $name" for (name, kernel) in (
            ("ContaminatedGaussian", cgkernel(0.6, 9.0, 0.7, 1)),)
        @testset "s = $s" for s in KERNEL_FD_COSTS
            c, g, H = robustifydkernel(kernel, s)
            gfd, Hfd = fdkernel(kernel, s)
            @test c ≈ robustify(kernel, s)
            @test g ≈ gfd rtol=1e-5
            @test H ≈ Hfd rtol=1e-3 atol=1e-5
            @test H == H'
        end
    end
end

@testset "adaptive kernel derivatives (finite differences)" begin
    res = SimpleResidual(0.7, 2)
    kernel = ContaminatedGaussian(0.5, 5.0, 0.4, static(1))
    mean = 0.1
    f(x) = NLLSModel.computecost(res, update(kernel, x, 1), mean + x[4])
    n = 4
    h = 1e-5
    e(i) = SVector{4, Float64}(ntuple(j -> j == i ? 1.0 : 0.0, 4))
    gfd = [(f(h * e(i)) - f(-h * e(i))) / 2h for i in 1:n]
    Hfd = [(f(h * e(i) + h * e(j)) - f(h * e(i) - h * e(j)) - f(-h * e(i) + h * e(j)) + f(-h * e(i) - h * e(j))) / 4h^2 for i in 1:n, j in 1:n]
    # Kernel parameters and variable both optimized
    c, g, H = NLLSModel.computecostgradhess(static(3), res, kernel, mean)
    @test c ≈ f(zeros(SVector{4, Float64}))
    @test g ≈ gfd rtol=1e-5
    @test H[1:3, 1:3] ≈ Hfd[1:3, 1:3] rtol=1e-3 atol=1e-5   # kernel-kernel block
    @test H[1:3, 4] ≈ Hfd[1:3, 4] rtol=1e-3 atol=1e-5       # cross block
    # Kernel only
    c, g, H = NLLSModel.computecostgradhess(static(1), res, kernel, mean)
    @test g ≈ gfd[1:3] rtol=1e-5
    @test H ≈ Hfd[1:3, 1:3] rtol=1e-3 atol=1e-5
end
