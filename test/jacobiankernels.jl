using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("jacobianproblems", "robustfit")

# R1: the kernel decides how each cost's rows enter the Jacobian system.

@testset "jacobiankernels.jl" begin
    # R1: per kernel, the Jacobian rows reproduce the Hessian linearization
    @testset "R1: J'J and J'r equal the Hessian and gradient, $name" for (name, kernel, wide, exact_hessian) in (
            ("NoRobust", NoRobust(), NoRobust(), true),
            ("Scaled", Scaled(NoRobust(), 2.5), Scaled(NoRobust(), 2.5), true),
            ("HuberKernel", HuberKernel(0.05), HuberKernel(1.0), true),
            ("Huber2oKernel", Huber2oKernel(0.05), Huber2oKernel(1.0), false),
            ("GemanMcclureKernel", GemanMcclureKernel(0.05), GemanMcclureKernel(1.0), true),
            ("GemanMcclure2oKernel", GemanMcclure2oKernel(0.05), GemanMcclure2oKernel(1.0), false))
        for (regime, outliers, kscale) in (("inliers only", 0, 1.0), ("with outliers", 4, 1.0))
            base = curvefit(outliers=outliers)
            # A wide kernel keeps the second-order correction unclamped when all residuals are small
            kern = regime == "inliers only" ? wide : kernel
            model = Model(Vector{Float64}, TlRob{TlExp, typeof(kern)})
            addvariable!(model, base.variables[1])
            for c in costlist(base)
                addcost!(model, TlRob(c, kern))
            end
            _, fdf!, xn, _ = optimvars(model; numthreads=static(1))
            g = zeros(3); H = zeros(3, 3)
            c = fdf!(g, H, xn)
            _, _, resjac!, xt, J = tlnls(model; numthreads=static(1), storage=:dense)
            r = zeros(size(J, 1))
            ct = resjac!(r, J, xt)
            @test ct ≈ c rtol=1e-12
            @test J' * r ≈ g rtol=1e-9 atol=1e-12
            if exact_hessian || regime == "inliers only"
                @test J' * J ≈ Symmetric(H) rtol=1e-9 atol=1e-12
            else
                # The Triggs correction is clamped for the outliers, where the two are not meant to agree: still PSD and finite
                @test all(isfinite, J)
                @test minimum(eigvals(Symmetric(J' * J))) > -1e-9
            end
            @test islinear(resjac!) == (kernel isa Union{NoRobust, Scaled})
        end
    end

    @testset "R1: a model mixing kernels reproduces the Hessian linearization" begin
        kernels = (NoRobust(), HuberKernel(0.5), Huber2oKernel(1.0), GemanMcclureKernel(0.5), GemanMcclure2oKernel(1.0), Scaled(NoRobust(), 2.0))
        base = curvefit()
        model = Model(Vector{Float64}, Union{[TlRob{TlExp, typeof(k)} for k in kernels]...})
        addvariable!(model, base.variables[1])
        for (i, c) in enumerate(costlist(base))
            addcost!(model, TlRob(c, kernels[mod1(i, length(kernels))]))
        end
        _, fdf!, xn, _ = optimvars(model; numthreads=static(1))
        g = zeros(3); H = zeros(3, 3)
        c = fdf!(g, H, xn)
        _, _, resjac!, xt, J = tlnls(model; numthreads=static(1), storage=:dense)
        r = zeros(size(J, 1))
        @test resjac!(r, J, xt) ≈ c rtol=1e-12
        @test J' * r ≈ g rtol=1e-9
        @test J' * J ≈ Symmetric(H) rtol=1e-9
        @test !islinear(resjac!)
    end

    @testset "R1: islinear" begin
        lin = curvefit()
        mixed = Model(Vector{Float64}, Union{TlExp, TlRob{TlExp, typeof(HuberKernel(0.05))}})
        addvariable!(mixed, lin.variables[1])
        for (i, c) in enumerate(costlist(lin))
            addcost!(mixed, i == 1 ? TlRob(c, HuberKernel(0.05)) : c)
        end
        robust = curvefit(huber=true)
        for (model, expected) in ((lin, true), (mixed, false), (robust, false))
            _, res!, resjac!, _, _ = tlnls(model; numthreads=static(1))
            @test islinear(res!) == expected
            @test islinear(resjac!) == expected
            @test islinear(flat_optimvars_nls(model)) == expected
            if expected
                @test NLLSModel.nls_callbacks(model) isa NamedTuple
            else
                @test_throws ArgumentError NLLSModel.nls_callbacks(model)
            end
        end
    end
end

@testset "alternation from a callback, Jacobian path, $(nameof(typeof(kernel))), $iterator, $nvar unfixed" for
        kernel in (ContaminatedGaussian(0.5, 5.0, 0.5, static(2)), GaussianPlusUniform(0.5, 0.5, static(2); volume=1600.0)),
        iterator in (:levenbergmarquardt, :dogleg), nvar in (2, 1)
    options = OptimizationOptions(; iterator, callbackchangesstate=true)
    model, truths = alternationmodel(kernel; nvar)
    result, given, before, returned = alternate!(model, options; formulation=:jacobian)
    @test isapprox(NLLSModel.params(model.variables[1]), alternationtruth(kernel); rtol=0.1)
    for (i, m) in enumerate(truths)
        @test isapprox(model.variables[i + 1], m; rtol=0.1)
    end
    @test !isempty(returned) && all(diff(returned) .<= 0)
    @test given ≈ before rtol=1e-12
    # The Hessian formulation ends at the same cost
    hmodel, _ = alternationmodel(kernel; nvar)
    alternate!(hmodel, options; formulation=:hessian)
    @test cost(model) ≈ cost(hmodel) rtol=1e-6
end

@testset "a callback that changes the cost needs callbackchangesstate, Jacobian path" begin
    model, _ = alternationmodel(ContaminatedGaussian(0.5, 5.0, 0.5, static(2)))
    err = try alternate!(model, OptimizationOptions(); formulation=:jacobian); nothing catch e e end
    @test err isa ArgumentError && occursin("callbackchangesstate", err.msg)
end

@testset "a residual's own volume on the Jacobian path" begin
    K = typeof(GaussianPlusUniform(0.8, 0.7, static(2); volume=400.0))
    rng = MersenneTwister(12)
    model = Model(Union{K, SVector{2, Float64}}, PairVolumeAdaptive{K})
    addvariable!(model, GaussianPlusUniform(0.8, 0.7, static(2); volume=400.0))
    addvariable!(model, SVector(0.3, -0.2))
    for _ in 1:50
        addcost!(model, PairVolumeAdaptive{K}(3 .* randn(rng, SVector{2, Float64}), log(400.0 * (0.5 + rand(rng))), SVector(1, 2)))
    end
    f, res!, resjac!, x0, J = optimvars_nls(model; unfixed=2, numthreads=static(1))
    r = zeros(NLLSModel.nresiduals(model))
    fh, = optimvars(model; unfixed=2, numthreads=static(1))
    @test resjac!(r, J, x0) ≈ fh(x0) rtol=1e-12
    @test resjac!(r, J, x0) ≈ cost(model) rtol=1e-12
end
