using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra
using SecondOrderOptimization, Krylov
import BlockLinearSystems: SchurBlockJacobian

include("fixtures/loader.jl")
fixture("jacobiancombinations")

# NTP4: the table of what `formulation = :jacobian` supports (the front door's docstring, `TALL_FORMULATION_DOC` in
# src/keywords.jl, the README and docs/src/index.md), as data. Every listed combination reaches the normal path's cost on a
# model with eliminable variables; every other one throws before any residual is evaluated. Which (prototype, solver,
# iterator) combinations work is the optimizer's table, tested there: this package never inspects `linsolver` or
# `iterator`, so each row runs every listed solver with :levenbergmarquardt and every iterator with :auto, and each
# unlisted solver once, with :levenbergmarquardt (rejection depends on the solver, never on the iterator).

@testset "jacobiancombinations.jl" begin
    model = countedchain()
    c0 = NLLSModel.cost(model)
    cn = NLLSModel.cost(first(optimize(model, OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200); numthreads=static(1))))

    @testset "reduction = $reduction, storage = $storage" for ((reduction, storage), (proto, linsolvers)) in TALL_TABLE
        @test optimvars_nls(model; reduction, storage, numthreads=static(1))[5] isa proto
        for linsolver in ALL_LINSOLVERS, iterator in TALL_ITERATORS
            listed = linsolver in linsolvers
            (iterator === :levenbergmarquardt || (listed && linsolver === :auto)) || continue
            TALL_EVALS[] = 0
            outcome = try
                options = OptimizationOptions(; iterator, linsolver, maxiters=200)
                solved, result = optimize(model, options; formulation=:jacobian, reduction, storage, numthreads=static(1))
                (NLLSModel.cost(solved), result)
            catch err
                err
            end
            if listed
                @test outcome isa Tuple
                outcome isa Tuple || continue
                c, result = outcome
                if iterator === :gradientdescent
                    # Gradient descent does not reach the minimum in 200 iterations (on either formulation it stalls about
                    # 3.5e-5 above it after some 17 000), so its cost is not compared with the normal path's: it must
                    # reduce the cost as the other iterators do
                    @test c < 0.1 * c0
                else
                    @test c ≈ cn rtol=(linsolver === :lsmr ? 1e-6 : 1e-8)
                end
                @test c ≈ result.bestcost rtol=1e-12
            else
                @test outcome isa ArgumentError
                @test TALL_EVALS[] == 0 # Thrown before any evaluation
            end
        end
    end

    # :auto picks a row of the table: elimination (the model has eliminable variables), and the normal path's reduced storage
    @testset ":auto is a row of the table" begin
        J = optimvars_nls(model; numthreads=static(1))[5]
        H = optimvars(model; numthreads=static(1))[4]
        @test J isa (H.R isa Matrix ? SchurBlockJacobian{Float64, Matrix{Float64}} : SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}})
        @test NLLSModel.cost(first(optimize(model, OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200); formulation=:jacobian, numthreads=static(1)))) ≈ cn rtol=1e-8
        @test cn < 0.1 * c0
    end
end
