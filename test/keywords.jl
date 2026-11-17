using NLLSModel, Test, Static, StaticArrays, LinearAlgebra, SparseArrays
using NLPModels, SecondOrderOptimization

# One keyword set for every way of building or optimizing a model (K1). Needs spymodel and offthread from
# tallformulation.jl.

const KW_OPTIONS = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=50)

# Every keyword of the set, at a valid value; each entry point takes them all
const KW_SET = (unfixed=nothing, numthreads=static(2), reduction=:none, storage=:auto, max_dense_bytes=10^8, min_work=100, min_work_cost=100)
const KW_ENTRY = (
    ("optimvars", (m; kw...) -> optimvars(m; kw...), KW_SET),
    ("optimvars_nls", (m; kw...) -> optimvars_nls(m; kw...), KW_SET),
    ("flat_optimvars_nls", (m; kw...) -> flat_optimvars_nls(m; kw...), KW_SET),
    ("optimize!, normal", (m; kw...) -> optimize!(deepcopy(m), KW_OPTIONS; kw...), KW_SET),
    ("optimize!, tall", (m; kw...) -> optimize!(deepcopy(m), KW_OPTIONS; formulation=:tall, kw...), KW_SET),
    ("NLPLSModel", (m; kw...) -> NLPLSModel(m; kw...), KW_SET),
    ("NLLSNLSModel", (m; kw...) -> NLLSNLSModel(m; kw...), KW_SET),
)

@testset "keywords.jl" begin
    model = spymodel(400, 2)

    @testset "every keyword of the set: $name" for (name, entry, kw) in KW_ENTRY
        @test entry(model; kw...) !== nothing
        # Old names are gone, not deprecated
        for old in (:reduced, :matrix_type, :min_work_per_thread)
            @test_throws MethodError entry(model; NamedTuple{(old,)}((old === :min_work_per_thread ? 100 : :auto,))...)
        end
    end

    @testset "unfixed and numthreads are not positional" begin
        @test_throws MethodError optimvars(model, nothing)
        @test_throws MethodError optimvars(model, nothing, static(1))
        @test_throws MethodError optimvars(model, 1)
    end

    @testset "unknown values are an ArgumentError" begin
        for (name, entry, _) in KW_ENTRY
            @test_throws ArgumentError entry(model; reduction=:schur)
            @test_throws ArgumentError entry(model; storage=:banded)
        end
    end

    # NTP2: the normal path's rule; with no eliminable variable :eliminate is the partition's ArgumentError (elimination on
    # the tall path itself is tested in tallelimination.jl)
    @testset "reduction on the tall path: no eliminable variable" begin
        err = try optimvars_nls(model; reduction=:eliminate); nothing catch e e end
        @test err isa ArgumentError && occursin("eliminable", err.msg)
        @test_throws ArgumentError flat_optimvars_nls(model; reduction=:eliminate)
        @test_throws ArgumentError NLLSNLSModel(model; reduction=:eliminate)
        x = [0.3, 0.1]
        r_auto, r_none = zeros(400), zeros(400)
        _, res_auto!, _, _, _ = optimvars_nls(model; reduction=:auto)
        _, res_none!, _, _, _ = optimvars_nls(model; reduction=:none)
        @test res_auto!(r_auto, x) == res_none!(r_none, x)
        @test r_auto == r_none
        # And from the front door, raised by optimvars_nls
        err = try optimize!(deepcopy(model), KW_OPTIONS; formulation=:tall, reduction=:eliminate); nothing catch e e end
        @test err isa ArgumentError && startswith(err.msg, "formulation = :tall:") && occursin("eliminable", err.msg)
    end

    # max_dense_bytes on the tall path: :auto switches to a sparse Jacobian just above the cap, :dense throws there
    @testset "max_dense_bytes on the tall path" begin
        Jof(; kw...) = optimvars_nls(model; numthreads=static(1), kw...)[5] # 400×2, 6400 bytes dense
        @test Jof(max_dense_bytes=6400) isa Matrix{Float64}
        @test Jof(max_dense_bytes=6399) isa SparseMatrixCSC
        @test Jof(storage=:dense, max_dense_bytes=6400) isa Matrix{Float64}
        err = try Jof(storage=:dense, max_dense_bytes=6399); nothing catch e e end
        @test err isa ArgumentError && occursin("400×2", err.msg)
        @test Jof(storage=:sparse) isa SparseMatrixCSC
        @test NLLSNLSModel(model; max_dense_bytes=6399).J_buf isa SparseMatrixCSC
        @test_throws ArgumentError NLLSNLSModel(model; storage=:dense, max_dense_bytes=6399)
    end

    @testset "NLPLSModel works on the full system" begin
        @test_throws ArgumentError NLPLSModel(model; reduction=:eliminate)
        @test NLPLSModel(model; reduction=:auto).fdf! isa NLLSModel.MultiVarDenseFDF
    end

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
