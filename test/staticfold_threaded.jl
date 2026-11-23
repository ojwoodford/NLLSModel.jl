using NLLSModel, StaticArrays, Static, LinearAlgebra, Test
include("fixtures/loader.jl")
fixture("allocations", "staticfold")

# The fold on a shared system (src/staticfold.jl): each task folds its chunk and writes it once into its own copy, and
# the copies are summed. Against the serial fold, and the in-place accumulation into a dynamic system.
@testset "staticfold.jl, threaded" begin
    nt = Threads.nthreads()
    @testset "$name" for (name, build, T, noalloc) in STATICFOLD_CASES
        model, unfixed = build()
        f1, fdf1!, x0, H1 = optimvars(model; unfixed, numthreads=static(1), precision=T)
        _, fdf!, _, H = optimvars(model; unfixed, numthreads=static(4), min_work=0, precision=T) # min_work = 0: shared
        ls = fdf!.linsystem
        @test ls isa NLLSModel.LinearSystemShared && ls isa NLLSModel.UniVariateLSstatic
        g1, g = zeros(MVector{size(H, 1), Float64}), zeros(MVector{size(H, 1), Float64})
        c1 = fdf1!(g1, H1, x0)
        c = fdf!(g, H, x0)
        # Threaded sums agree with serial to rounding, not bit for bit
        @test isapprox(c, c1; rtol=1e-13)
        @test isapprox(g, g1; rtol=1e-12)
        @test isapprox(H, H1; rtol=T === Float32 ? 8 * eps(Float32) : 1e-12) # a Float32 sum's order shows at its rounding
        @test H == H' # each copy holds both triangles, so the sum is symmetric with no mirror
        noalloc && @test threaded_allocs_ok(allocs_warm(fdf!, g, H, x0), nt > 1)
        @test (@inferred NLLSModel.staticfold!(g, H, ls, model.variables, model.costs)) == c
        # A Hessian not of the system's type: the sum lands in the system's own copies, then is copied
        gf, Hf = zeros(length(g)), zeros(T, size(H))
        @test fdf!(gf, Hf, x0) == c
        @test gf == g && Hf == H
        @test NLLSModel.getA(ls) == H
    end

    @testset "a shared system with one copy" begin
        model, _ = scalarmodel(3, 50)
        _, fdf!, x0, H = optimvars(model; numthreads=static(1))
        ls1 = fdf!.linsystem
        shared1 = NLLSModel.LinearSystemShared(ls1.x, SVector{1}(ls1.ls), MVector{1, Float64}(undef))
        g = zero(H[:, 1])
        c = fdf!(g, H, x0)
        g1, H1 = zero(g), zero(H)
        @test NLLSModel.staticfold!(g1, H1, shared1, model.variables, model.costs) == c
        @test g1 == g && H1 == H
    end
end
