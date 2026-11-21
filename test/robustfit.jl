using NLLSModel, Test, Static, StaticArrays, Random

include("fixtures/loader.jl")
fixture("robustfit")

@testset "robustfit.jl" begin
    @testset "recovery, ContaminatedGaussian, d = $d" for d in 1:2
        s = mixturesqnorms(MersenneTwister(2), 20_000, d, 1.0, 10.0, 0.8)
        p = NLLSModel.params(optimizerobustifier(cgkernel(0.5, 5.0, 0.5, d), s; maxiters=100))
        @test p[1] ≈ 1.0 rtol=0.03
        @test p[2] ≈ 10.0 rtol=0.03
        @test p[3] ≈ 0.8 rtol=0.03
    end

    @testset "recovery, GaussianPlusUniform, d = $d" for d in 1:2
        side = 40.0
        s = uniformsqnorms(MersenneTwister(3), 20_000, d, 1.0, 0.8, side)
        p = NLLSModel.params(optimizerobustifier(gpukernel(0.5, 0.5, d; volume=side^d), s; maxiters=100))
        @test p[1] ≈ 1.0 rtol=0.03
        @test p[2] ≈ 0.8 rtol=0.03
    end

    @testset "an iteration does not increase the cost plus the prior's term, $name, wprior = $wprior" for (name, kernel, s, logvolume) in (
            ("ContaminatedGaussian", cgkernel(0.3, 3.0, 0.5, 2), mixturesqnorms(MersenneTwister(4), 500, 2, 1.0, 10.0, 0.8), nothing),
            ("GaussianPlusUniform", gpukernel(0.3, 0.5, 2; volume=400.0), uniformsqnorms(MersenneTwister(5), 500, 2, 1.0, 0.8, 20.0), nothing),
            ("GaussianPlusUniform, per-residual volumes", gpukernel(0.3, 0.5, 2; volume=400.0),
                uniformsqnorms(MersenneTwister(5), 500, 2, 1.0, 0.8, 20.0), log.(400.0 .* (0.5 .+ rand(MersenneTwister(6), 500))))),
            wprior in ((1.0, 1.0), (5.0, 2.0))
        k = kernel
        obj = fitobjective(k, s, logvolume; wprior)
        for _ in 1:30
            k = isnothing(logvolume) ? optimizerobustifier(k, s; maxiters=1, wprior) : optimizerobustifier(k, s, logvolume; maxiters=1, wprior)
            newobj = fitobjective(k, s, logvolume; wprior)
            @test newobj <= obj + 1e-12 * abs(obj) # Rounding only
            obj = newobj
        end
    end

    @testset "wprior, $name" for (name, kernel) in (("ContaminatedGaussian", cgkernel(0.5, 5.0, 0.5, 1)), ("GaussianPlusUniform", gpukernel(0.5, 0.5, 1; volume=40.0)))
        s = mixturesqnorms(MersenneTwister(7), 10, 1, 1.0, 10.0, 0.8)
        w0 = NLLSModel.params(optimizerobustifier(kernel, s))[end]
        wp = NLLSModel.params(optimizerobustifier(kernel, s; wprior=(30.0, 2.0)))[end]
        mode = (30.0 - 1) / (30.0 + 2.0 - 2)
        @test abs(wp - mode) < abs(w0 - mode)
        @test optimizerobustifier(kernel, s; wprior=(1, 1)) == optimizerobustifier(kernel, s)
        for bad in ((0.5, 1.0), (1.0, 0.99), (Inf, 1.0), (1.0, NaN))
            @test_throws ArgumentError optimizerobustifier(kernel, s; wprior=bad)
        end
    end

    @testset "per-residual log volumes" begin
        k = gpukernel(0.5, 0.5, 2; volume=400.0)
        s = uniformsqnorms(MersenneTwister(8), 200, 2, 1.0, 0.8, 20.0)
        @test optimizerobustifier(k, s, fill(k.logvolume, 200)) == optimizerobustifier(k, s)
        @test optimizerobustifier(k, s, log.(range(10.0, 4000.0; length=200))) != optimizerobustifier(k, s)
        @test_throws DimensionMismatch optimizerobustifier(k, s, zeros(199))
        @test_throws MethodError optimizerobustifier(cgkernel(0.5, 5.0, 0.4, 2), s, zeros(200))
    end

    @testset "degenerate input, $name" for (name, kernel) in (("ContaminatedGaussian", cgkernel(0.5, 5.0, 0.4, 1)), ("GaussianPlusUniform", gpukernel(0.5, 0.4, 1; volume=40.0)))
        for s in (Float64[], zeros(10), fill(1e300, 10), [2.0])
            @test all(isfinite, NLLSModel.params(optimizerobustifier(kernel, s)))
        end
        @test optimizerobustifier(kernel, Float64[]) === kernel
    end

    @testset "optimizerobustifier!" begin
        K = typeof(gpukernel(0.5, 0.5, 2; volume=400.0))
        CGK = typeof(cgkernel(0.5, 5.0, 0.5, 2))
        rng = MersenneTwister(9)
        data = [rand(rng) < 0.8 ? randn(rng, SVector{2, Float64}) : 20 .* (rand(rng, SVector{2, Float64}) .- 0.5) for _ in 1:300]
        logvolumes = log.(400.0 .* (0.5 .+ rand(rng, 300)))
        x = SVector(0.1, -0.2)
        s_hand = [sum(abs2, x - y) for y in data]

        model = Model(Union{K, SVector{2, Float64}}, Union{PairVolumeAdaptive{K}, PairFixed{K}})
        addvariable!(model, x)                                    # 1
        addvariable!(model, gpukernel(0.5, 0.5, 2; volume=400.0)) # 2: fitted
        addvariable!(model, gpukernel(2.0, 0.3, 2; volume=400.0)) # 3: another kernel, used
        addvariable!(model, gpukernel(0.7, 0.6, 2; volume=400.0)) # 4: a kernel no residual uses
        for (y, l) in zip(data, logvolumes)
            addcost!(model, PairVolumeAdaptive{K}(y, l, SVector(2, 1)))
            addcost!(model, PairVolumeAdaptive{K}(y, l, SVector(3, 1)))
            addcost!(model, PairFixed(y, gpukernel(0.5, 0.5, 2; volume=400.0))) # A fixed kernel: not visited
        end
        start = model.variables[2]
        @test optimizerobustifier(start, s_hand, logvolumes) != optimizerobustifier(start, s_hand) # The residuals' own volumes matter
        other, unused = model.variables[3], model.variables[4]
        s, lv = Float64[], Float64[]
        c = optimizerobustifier!(model, 2; s, logvolume=lv)
        # It collects the residuals' squared norms (to rounding: the order of the sum depends on how it is compiled) and
        # their own log volumes, fits with them, and writes the kernel back
        @test s ≈ s_hand rtol=1e-14
        @test lv == logvolumes
        @test model.variables[2] == optimizerobustifier(start, s, lv)
        @test c == cost(model)
        @test model.variables[3] === other && model.variables[4] === unused # Only the kernel asked for
        # A kernel no residual uses is unchanged, and the cost returned
        @test optimizerobustifier!(model, 4) == cost(model)
        @test model.variables[4] === unused
        # An index that is not one of the two kernels
        @test_throws ArgumentError optimizerobustifier!(model, 1)
        err = try optimizerobustifier!(model, 1); nothing catch e e end
        @test occursin("ContaminatedGaussian", err.msg) && occursin("GaussianPlusUniform", err.msg)
        @test_throws ArgumentError optimizerobustifier!(model, 2; wprior=(0.0, 1.0))
        # A second call with the same buffers allocates nothing
        s, lv = Float64[], Float64[]
        @test allocs_warm(fitmodel, model, 2, s, lv) == 0

        # A ContaminatedGaussian: logvolume is not touched
        cgmodel = Model(Union{CGK, SVector{2, Float64}}, PairAdaptive{CGK})
        addvariable!(cgmodel, cgkernel(0.5, 5.0, 0.5, 2))
        addvariable!(cgmodel, x)
        for y in data
            addcost!(cgmodel, PairAdaptive{CGK}(y))
        end
        startcg = cgmodel.variables[1]
        lv = [1.0]
        @test optimizerobustifier!(cgmodel, 1; s, logvolume=lv) == cost(cgmodel)
        @test s ≈ s_hand rtol=1e-14
        @test cgmodel.variables[1] == optimizerobustifier(startcg, s) && lv == [1.0]
        @test allocs_warm(fitmodel, cgmodel, 1, s, lv) == 0
    end

    @testset "the docstring example of optimizerobustifier" begin
        s = [0.1, 0.5, 2.0, 0.3, 80.0, 0.7, 1.2, 150.0]
        kernel = optimizerobustifier(ContaminatedGaussian(0.5, 5.0, 0.5, static(1)), s)
        @test kernel isa ContaminatedGaussian{Float64, 1} && all(isfinite, NLLSModel.params(kernel))
    end

    @testset "the low-level fit allocates nothing" begin
        s = uniformsqnorms(MersenneTwister(10), 100, 2, 1.0, 0.8, 20.0)
        @test allocs_warm(fitlow, gpukernel(0.5, 0.5, 2; volume=400.0), s) == 0
        @test allocs_warm(fitlow, gpukernel(0.5, 0.5, 2; volume=400.0), s, fill(log(400.0), 100)) == 0
        @test allocs_warm(fitlow, cgkernel(0.5, 5.0, 0.5, 2), s) == 0
    end
end
