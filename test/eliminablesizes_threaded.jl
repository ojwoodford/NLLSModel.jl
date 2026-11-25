using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems: SchurBlockHessian, SchurBlockJacobian
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("allocations", "eliminablesizes")

# The testsets of eliminablesizes.jl whose behaviour depends on the thread count (the `threaded` group, run on 1 and on 4
# threads).
@testset "eliminablesizes.jl, threaded" begin
    @testset "eliminated variables of $N dimensions" for N in (1, 8, 9)
        model = sizedmodel(N)
        # Zero allocation per evaluation, threaded (rules.md 21's 1.10 bound for a threaded call)
        nt = Threads.nthreads()
        for storage in (:dense, :sparse)
            a = hessianallocs(model; numthreads=static(nt), reduction=:eliminate, storage, min_work=0, min_work_cost=0)
            @test threaded_allocs_ok(a[1], a[3]) && threaded_allocs_ok(a[2], a[4])
            @test warmallocs(model; reduction=:eliminate, storage, numthreads=static(nt), min_work=0) == (0, 0, 0)
        end
    end

    @testset "a cost added after the thread plan was made: evaluated on one thread" begin
        nt = Threads.nthreads()
        model = sizedmodel(1)
        _, fdf!, x0, H = optimvars(model; numthreads=static(nt), reduction=:eliminate, storage=:dense, min_work=0, min_work_cost=0)
        @test H isa SchurBlockHessian
        @test (fdf!.plan !== nothing) == (nt > 1)
        # A second prior on the first pose: a new cost, but no new block of the Hessian
        addcost!(model, SizedPrior(model.variables[1] .+ 0.1, 1))
        g = zeros(nvars(x0))
        c = fdf!(g, H, x0)
        _, fdf1!, x01, H1 = optimvars(model; numthreads=static(1), reduction=:eliminate, storage=:dense)
        g1 = zeros(nvars(x01))
        @test c ≈ fdf1!(g1, H1, x01) rtol=1e-12
        @test g ≈ g1 rtol=1e-12
    end
end
