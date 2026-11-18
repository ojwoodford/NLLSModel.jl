using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems: SchurBlockHessian, SchurBlockJacobian
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("allocations", "eliminablesizes")

# E1: eliminable variables of any size BlockLinearSystems takes as a block, on the Hessian and the Jacobian path (256
# dimensions: bigvariable.jl).

@testset "eliminablesizes.jl" begin
    opts = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200)
    solvedcost(model; kwargs...) = NLLSModel.cost(first(optimize(model, opts; numthreads=static(1), kwargs...)))
    # One size per tier BlockLinearSystems' kernels have: closed form (1 to 3), static (to 8, the last), dynamic (9, the
    # first). BlockLinearSystems tests every size at kernel level; this package has no threshold of its own in this range
    @testset "eliminated variables of $N dimensions" for N in (1, 8, 9)
        model = sizedmodel(N)
        c0 = NLLSModel.cost(model)
        cn = solvedcost(model; reduction=:none)
        @test cn < 0.01 * c0
        # :auto eliminates, as :eliminate does
        @test optimvars(model; numthreads=static(1))[4] isa SchurBlockHessian
        @test optimvars_nls(model; numthreads=static(1))[5] isa SchurBlockJacobian
        for storage in (:dense, :sparse)
            @test optimvars(model; numthreads=static(1), reduction=:eliminate, storage)[4] isa SchurBlockHessian
            @test solvedcost(model; reduction=:eliminate, storage) ≈ cn rtol=1e-8
            @test solvedcost(model; formulation=:jacobian, reduction=:eliminate, storage) ≈ cn rtol=1e-8
        end
        # Zero allocation per evaluation, serial (threaded: eliminablesizes_threaded.jl)
        for storage in (:dense, :sparse)
            @test hessianallocs(model; numthreads=static(1), reduction=:eliminate, storage)[1:2] == (0, 0)
            @test warmallocs(model; reduction=:eliminate, storage, numthreads=static(1)) == (0, 0, 0)
        end
    end
end
