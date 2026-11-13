using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
import OptimizationVariableInterface

# R1: the kernel decides how each cost's rows enter the tall system. Needs tallproblems.jl.

@testset "tallkernels.jl" begin
    # R1: per kernel, the tall rows reproduce the normal-equations linearization
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
            _, fdf!, xn, _ = optimvars(model, nothing, static(1))
            g = zeros(3); H = zeros(3, 3)
            c = fdf!(g, H, xn)
            _, _, resjac!, xt, J = tlnls(model; numthreads=static(1), matrix_type=:dense)
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

    @testset "R1: a model mixing kernels reproduces the normal-equations linearization" begin
        kernels = (NoRobust(), HuberKernel(0.5), Huber2oKernel(1.0), GemanMcclureKernel(0.5), GemanMcclure2oKernel(1.0), Scaled(NoRobust(), 2.0))
        base = curvefit()
        model = Model(Vector{Float64}, Union{[TlRob{TlExp, typeof(k)} for k in kernels]...})
        addvariable!(model, base.variables[1])
        for (i, c) in enumerate(costlist(base))
            addcost!(model, TlRob(c, kernels[mod1(i, length(kernels))]))
        end
        _, fdf!, xn, _ = optimvars(model, nothing, static(1))
        g = zeros(3); H = zeros(3, 3)
        c = fdf!(g, H, xn)
        _, _, resjac!, xt, J = tlnls(model; numthreads=static(1), matrix_type=:dense)
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
                @test nls_callbacks(model) isa NamedTuple
            else
                @test_throws ArgumentError nls_callbacks(model)
            end
        end
    end
end
