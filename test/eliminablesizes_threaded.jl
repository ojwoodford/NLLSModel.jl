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
            a = normalallocs(model; numthreads=static(nt), reduction=:eliminate, storage, min_work=0, min_work_cost=0)
            @test threaded_allocs_ok(a[1], a[3]) && threaded_allocs_ok(a[2], a[4])
            @test warmallocs(model; reduction=:eliminate, storage, numthreads=static(nt), min_work=0) == (0, 0, 0)
        end
    end
end
