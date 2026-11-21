using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra, Random
import SecondOrderOptimization
import SecondOrderOptimization: optimize!, OptimizationOptions, HessianObjective
import OptimizationVariableInterface, BlockLinearSystems

include("fixtures/loader.jl")
fixture("bigvariable")

# Variables of any size. A variable that becomes a block of a block linear system is limited only by BlockLinearSystems,
# which checks block sizes itself (this package states no limit); a variable optimized on its own, or on the plain
# residual-and-Jacobian path, has none.

@testset "bigvariable.jl" begin
    # A variable of any size can be added
    problem = bigproblem(256)
    @test length(problem.variables[1]) == 256

    # It can be optimized on its own, and the single variable system does not use blocks
    f, fdf!, x0, H = optimvars(problem; unfixed=1, numthreads=static(1))
    @test H isa Matrix{Float64} && size(H) == (256, 256)
    res = optimize!(x0, HessianObjective(f, fdf!; H), OptimizationOptions(iterator=:levenbergmarquardt, maxiters=20))
    x = x0
    @test f(x) < 1e-8
    @test x ≈ collect(1.0:256)

    # The residual-and-Jacobian path writes a flat Jacobian, so it accepts a larger variable too
    f, res!, resjac!, x0, J = optimvars_nls(problem; unfixed=1, numthreads=static(1))
    r = zeros(size(J, 1))
    Jm = copy(J)
    @test resjac!(r, Jm, x0) ≈ f(x0)
    @test size(J, 2) == 256

    # A variable of 256 dimensions (past the old limit of 255) is a block: retained, in the plain system (dense and sparse) and
    # in a Schur system, and eliminated; every route reaches the same cost. The variables are Vectors, so any size past the
    # limit runs the same code
    opts = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=100)
    solved(problem; kwargs...) = NLLSModel.cost(first(SecondOrderOptimization.optimize(problem, opts; numthreads=static(1), kwargs...)))
    @testset "a retained block of $n dimensions" for n in (256,)
        problem = bigproblem(n)
        problem.costs[PinRes][2] = PinRes([1.5, 2.0, 3.0], 2) # In conflict with the link to variable 1, so the minimum is not zero
        c0 = NLLSModel.cost(problem)
        cn = solved(problem; reduction=:none, storage=:dense)
        @test 0 < cn < 0.1 * c0
        @test optimvars(problem; numthreads=static(1), storage=:sparse)[4] isa SparseMatrixCSC
        @test solved(problem; reduction=:none, storage=:sparse) ≈ cn rtol=1e-8
        @test solved(problem; formulation=:jacobian, storage=:dense) ≈ cn rtol=1e-8
        @test solved(problem; formulation=:jacobian, storage=:sparse) ≈ cn rtol=1e-8
        # Retained in a Schur system
        schur = schurproblem(n)
        @test optimvars(schur; numthreads=static(1))[4] isa BlockLinearSystems.SchurBlockHessian
        for storage in (:dense, :sparse)
            @test solved(schur; reduction=:eliminate, storage) ≈ solved(schur; reduction=:none) atol=1e-12
            @test solved(schur; formulation=:jacobian, reduction=:eliminate, storage) ≈ solved(schur; reduction=:none) atol=1e-12
        end
    end
    @testset "eliminated blocks of $n dimensions" for n in (256,)
        problem = bigelimproblem(n)
        c0 = NLLSModel.cost(problem)
        cn = solved(problem; reduction=:none)
        @test cn < 0.01 * c0
        @test optimvars(problem; numthreads=static(1))[4] isa BlockLinearSystems.SchurBlockHessian
        for storage in (:dense, :sparse)
            @test solved(problem; reduction=:eliminate, storage) ≈ cn rtol=1e-8
            @test solved(problem; formulation=:jacobian, reduction=:eliminate, storage) ≈ cn rtol=1e-8
        end
    end
end
