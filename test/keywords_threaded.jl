using NLLSModel, Test, Static, StaticArrays, LinearAlgebra, SparseArrays
using NLPModels, SecondOrderOptimization

include("fixtures/loader.jl")
fixture("keywords", "tallformulation")

# The testsets of keywords.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4 threads).
@testset "keywords.jl, threaded" begin
    model = spymodel(400, 2)

    # The two NLPModels wrappers pass the floors on: a floor of 0 threads, a huge floor does not
    @testset "the NLPModels wrappers pass the floors on" begin
        x = [0.3, 0.1]
        nlp(floor) = NLPLSModel(model; numthreads=static(4), min_work=floor, min_work_cost=floor)
        nls(floor) = NLLSNLSModel(model; numthreads=static(4), min_work=floor, min_work_cost=floor)
        @test obj(nlp(0), x) ≈ obj(nlp(10^9), x) rtol=1e-13
        @test grad(nlp(0), x) ≈ grad(nlp(10^9), x) rtol=1e-12
        @test obj(nls(0), x) ≈ obj(nls(10^9), x) rtol=1e-13
        @test residual(nls(0), x) == residual(nls(10^9), x)
        if Threads.nthreads() > 1
            for (make, eval) in ((nlp, m -> obj(m, x)), (nlp, m -> grad(m, x)), (nls, m -> obj(m, x)), (nls, m -> residual(m, x)))
                @test offthread(() -> eval(make(0))) && !offthread(() -> eval(make(10^9)))
            end
        end
        # obj alone evaluates the cost function, not the residual
        m = nls(10^9)
        obj(m, x)
        @test !m.res_valid
        residual(m, x)
        @test m.res_valid
        @test obj(m, x) ≈ 0.5 * sum(abs2, residual(m, x)) rtol=1e-13
    end

    @testset "numthreads defaults to every thread everywhere" begin
        nt = Threads.nthreads()
        decided(f) = f.nthreads == NLLSModel.threadcount(f.work, f.min_work, nt, NLLSModel.costsplit(model.costs))
        @test decided(NLPLSModel(model; min_work_cost=1).f) && NLPLSModel(model; min_work_cost=1).f.nthreads == nt
        @test decided(optimvars(model; min_work_cost=1)[1]) && optimvars(model; min_work_cost=1)[1].nthreads == nt
        res! = optimvars_nls(model; min_work=1)[2]
        @test res!.nthreads == NLLSModel.threadcount(res!.layout.work, 1, nt, NLLSModel.costsplit(model.costs)) == nt
    end
end
