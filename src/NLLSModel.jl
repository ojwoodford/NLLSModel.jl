"""
    NLLSModel

Nonlinear least-squares (and general smooth cost) problems as a bipartite graph of variables and costs: build a
[`Model`](@ref), and evaluate its cost, gradient and Gauss-Newton Hessian ([`optimvars`](@ref)) or its residual and
Jacobian ([`optimvars_nls`](@ref)) for an optimizer. With SecondOrderOptimization loaded, `optimize!(model, options)`
solves it directly.
"""
module NLLSModel

using BlockLinearSystems
using FastNorm
using OptimizationVariableInterface
using SIRBSDispatch
using Polyester: @batch
using LoopVectorization: @turbo
import ForwardDiff, DiffResults
using Static, StaticArrays, SparseArrays, LinearAlgebra

import BlockLinearSystems:
    SparseBlockLinearSystem, DenseBlockLinearSystem, TransposedPattern,
    block, zero!, copy_structure,
    symmetrifyfull, symmetrifyfull!, sparse_with_map, copyto!,
    block_sparse_nnz, computestartindices,
    SchurStructure, SchurBlockHessian, build_schur_pattern,
    threadargs, fromthreadargs

import OptimizationVariableInterface:
    nvars, update, syncvars!,
    ZeroToInfScalar, ZeroToOneScalar

import FastNorm: sqnorm
import SIRBSDispatch: sirbsdispatch
using DampedLinearSolve: DampedLinearSolve, makehessiansystem, dampen!, solve!, getdiag!, isfinitesystem, SolveStatus, Success

# Export NLLSModel public interface
# Problem construction & evaluation
export Model, NLPLSModel, NLLSNLSModel
export addvariable!, addcost!, subproblem, subproblem!
export cost, optimvars, optimvars_nls, flat_optimvars_nls, syncvars!
export islinear

# Choosing how to solve a model
export recommendoptimizer, OptimizationRecommendation

# Cost and residual types & definition interface
export AbstractCost, AbstractResidual, AbstractAdaptiveResidual
export ndeps, nres, varindices, getvars
export computeresidual, computeresjac, computecost, computecostgradhess

# Built-in error residuals
export SimpleError2, SimpleError3, SimpleError4, generatemeasurement

# Robust cost functions & interface
export AbstractRobustifier, AbstractAdaptiveRobustifier
export NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel, ContaminatedGaussian
export GaussianPlusUniform, withlogvolume
export robustkernel, robustify, robustifydcost, robustifydkernel

# Exported abstract types
abstract type AbstractCost end  # Standard (non-squared) cost
abstract type AbstractResidual <: AbstractCost end # Squared (or robustified squared) cost
abstract type AbstractAdaptiveResidual <: AbstractResidual end # Squared cost with adaptive robustifier
abstract type AbstractRobustifier end # Robustifier with fixed parameters
abstract type AbstractAdaptiveRobustifier <: AbstractRobustifier end # Robustifier with variable parameters

# Constants
const MAX_ARGS = 10 # Maximum number of variables a residual can depend on
const MAX_BLOCK_SZ = 32 # Maximum DoF of a variable and also maximum length of a residual allowed for various operations, such as marginalization
"""
    MAX_STATIC_VAR

Maximum total size of the unfixed variables of a cost for static-sized autodiff (duals of static width), and so for the
full static path (that autodiff and the static multiply of `gramgrad`). Above it, or with a run-time size, the cost is
differentiated by ForwardDiff with a configuration cached per task. Inherited from NLLSsolver; not measured.
"""
const MAX_STATIC_VAR = 64

"""
    MAX_STATIC_JAC

Maximum rows × columns (`nres` times the total size of the unfixed variables) of the Jacobian of a cost for the full
static path. It bounds the size of the static Jacobian that is built and unrolled. A cost above it (with static sizes, at
most `MAX_STATIC_VAR` in all) is still differentiated with duals of static width, its Jacobian written into a `Matrix`
buffer per task, and takes the static-column or `@turbo` kernel. A starting value, not measured.
"""
const MAX_STATIC_JAC = 128

"""
    MAX_STATIC_ROWS

Maximum static residual length (`nres` returning a `StaticInt`) for the static-column kernel of `gramgrad!`, which is
used for costs that are not on the full static path. Longer (or run-time) residuals use the `@turbo` kernel. Tested
independently of the number of columns. A starting value, not measured.
"""
const MAX_STATIC_ROWS = 16

# Includes
include("utils.jl")
include("CostStruct.jl")
include("problem.jl")
include("unroll.jl")
include("linearsystem.jl")
include("robust.jl")
include("robustadaptive.jl")
include("gramkernels.jl")
include("residual.jl")
include("autodiff.jl")
include("cost.jl")
include("multithreading.jl")
include("schurthreading.jl")
include("keywords.jl")
include("flatvars.jl")
include("optimize.jl")
include("jacobian_system.jl")
include("recommender.jl")
include("docstrings.jl")

end # module NLLSModel
