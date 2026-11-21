using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import BlockLinearSystems
import OptimizationVariableInterface

fixture("optimizeba")

# Eliminable landmark (for the Schur solvers)
struct OptLandmark{T}
    p::SVector{3, T}
end
OptimizationVariableInterface.nvars(::OptLandmark) = static(3)
OptimizationVariableInterface.update(v::OptLandmark, step, offset=1) = OptLandmark(v.p + SVector(step[offset], step[offset+1], step[offset+2]))
OptimizationVariableInterface.iseliminable(::Type{<:OptLandmark}) = static(true)
NLLSModel.generatemeasurement(camera, pt::OptLandmark) = NLLSModel.generatemeasurement(camera, pt.p)

function optlandmark_problem(ncams, nlms)
    problem = Model(Union{SVector{6, Float64}, OptLandmark{Float64}}, SimpleError2{2, Float64, SVector{6, Float64}, OptLandmark{Float64}})
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i in 1:ncams
        addvariable!(problem, randn(SVector{6, Float64}) .+ camoffset)
    end
    for i in 1:nlms
        addvariable!(problem, OptLandmark{Float64}(rand(SVector{3, Float64}) .+ SVector(-0.5, -0.5, 10.0)))
    end
    for c in 1:ncams, l in 1:nlms
        if (c + l) % 2 == 0 || c == 1
            addcost!(problem, SimpleError2{SVector{6, Float64}, OptLandmark{Float64}}(NLLSModel.generatemeasurement(problem.variables[c]::SVector{6, Float64}, problem.variables[ncams + l]), c, ncams + l))
        end
    end
    # Perturb the landmarks
    for i in ncams+1:ncams+nlms
        problem.variables[i] = OptLandmark{Float64}(problem.variables[i].p + 0.01 * randn(SVector{3, Float64}))
    end
    return problem
end

function solve(problem; iterator=:levenbergmarquardt, reduction=:auto, storage=:auto, maxiters=100)
    c0 = NLLSModel.cost(problem)
    res = optimize!(problem, OptimizationOptions(iterator=iterator, maxiters=maxiters); numthreads=static(1), reduction, storage)
    return c0, NLLSModel.cost(problem), res
end
