using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra
using SecondOrderOptimization, Krylov
import BlockLinearSystems: SchurBlockJacobian

# NTP4: the table of what `formulation = :tall` supports (the front door's docstring, `TALL_FORMULATION_DOC` in
# src/keywords.jl, the README and docs/src/index.md), as data. Every listed combination reaches the normal path's cost on a
# model with eliminable variables; every other one throws before any residual is evaluated. Needs posechain and its costs
# from tallelimination.jl.

# (reduction, storage) => (the Jacobian prototype, the linsolver values the optimizer accepts with it)
const TALL_TABLE = Dict(
    (:none, :dense) => (Matrix{Float64}, (:auto, :qr, :lsmr)),
    (:none, :sparse) => (SparseMatrixCSC{Float64, Int}, (:auto, :qr, :lsmr)),
    (:eliminate, :dense) => (SchurBlockJacobian{Float64, Matrix{Float64}}, (:auto, :qr, :cholesky, :ldl, :lsmr)),
    (:eliminate, :sparse) => (SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}, (:auto, :qr, :cholesky, :ldl, :lsmr)),
)
const TALL_ITERATORS = (:levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg, :newton)
# Every linsolver symbol an extension accepts, and every iterator
const ALL_LINSOLVERS = (:auto, :cholesky, :ldl, :qr, :cg, :minres, :krylov, :lsmr, :pardiso)
const ALL_ITERATORS = (TALL_ITERATORS..., :gradientdescent)

# A cost that counts its evaluations (the residual, and the residual under autodiff)
const TALL_EVALS = Threads.Atomic{Int}(0)
struct Counted{C} <: NLLSModel.AbstractResidual
    c::C
end
NLLSModel.ndeps(r::Counted) = NLLSModel.ndeps(r.c)
NLLSModel.nres(r::Counted) = NLLSModel.nres(r.c)
NLLSModel.varindices(r::Counted) = NLLSModel.varindices(r.c)
NLLSModel.getvars(r::Counted, vars::AbstractVector) = NLLSModel.getvars(r.c, vars)
NLLSModel.computeresidual(r::Counted, args...) = (Threads.atomic_add!(TALL_EVALS, 1); NLLSModel.computeresidual(r.c, args...))
Base.eltype(::Counted) = Float64

function countedchain()
    base = posechain(8, 12)
    model = Model(eltype(base.variables), Union{Counted{ChainPrior}, Counted{ChainOdometry}, Counted{ChainSighting}})
    foreach(v -> addvariable!(model, v), base.variables)
    foreach(costvec -> foreach(c -> addcost!(model, Counted(c)), costvec), values(base.costs))
    return model
end

@testset "tallcombinations.jl" begin
    model = countedchain()
    c0 = NLLSModel.cost(model)
    cn = NLLSModel.cost(first(optimize(model, OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200); numthreads=static(1))))

    @testset "reduction = $reduction, storage = $storage" for ((reduction, storage), (proto, linsolvers)) in TALL_TABLE
        @test optimvars_nls(model; reduction, storage, numthreads=static(1))[5] isa proto
        for linsolver in ALL_LINSOLVERS, iterator in ALL_ITERATORS
            listed = linsolver in linsolvers && iterator in TALL_ITERATORS
            TALL_EVALS[] = 0
            outcome = try
                options = OptimizationOptions(; iterator, linsolver, maxiters=200)
                solved, result = optimize(model, options; formulation=:tall, reduction, storage, numthreads=static(1))
                (NLLSModel.cost(solved), result)
            catch err
                err
            end
            if listed
                @test outcome isa Tuple
                outcome isa Tuple || continue
                c, result = outcome
                @test c ≈ cn rtol=(linsolver === :lsmr ? 1e-6 : 1e-8)
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
        @test NLLSModel.cost(first(optimize(model, OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200); formulation=:tall, numthreads=static(1)))) ≈ cn rtol=1e-8
        @test cn < 0.1 * c0
    end
end
