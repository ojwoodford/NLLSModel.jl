module NLLSModel

using BlockMatrices
using FastNorm
using OptimizationVariableInterface
using ValueDispatch
using Static, StaticArrays, SparseArrays, LinearAlgebra

import BlockMatrices:
    BlockSparseMatrix, BlockDenseMatrix,
    block, zero!, copy_structure,
    symmetrifyfull, makesparseindices,
    sparse_dense_decision, computestartindices, find_diag_inds

import OptimizationVariableInterface:
    nvars, update, update!, syncvars!,
    ZeroToInfScalar, ZeroToOneScalar

import FastNorm: sqnorm
import ValueDispatch: valuedispatch

# Export NLLSModel public interface
# Problem construction & evaluation
export Model, NLPLSModel
export addvariable!, addcost!, addresidual!, subproblem, subproblem!
export cost, optimvars, syncvars!

# Cost and residual types & definition interface
export AbstractCost, AbstractResidual, AbstractAdaptiveResidual
export ndeps, nres, varindices, getvars
export computeresidual, computeresjac, computecost, computecostgradhess

# Built-in error residuals
export SimpleError2, SimpleError3, SimpleError4, generatemeasurement

# Robust cost functions & interface
export AbstractRobustifier, AbstractAdaptiveRobustifier
export NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel, ContaminatedGaussian
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
const MAX_STATIC_VAR = 64 # Maximum static variable size for static sized autodiff to be used

# Includes
include("utils.jl")
include("CostStruct.jl")
include("problem.jl")
const addresidual! = addcost!
include("unroll.jl")
include("linearsystem.jl")
include("robust.jl")
include("robustadaptive.jl")
include("residual.jl")
include("autodiff.jl")
include("cost.jl")
include("multithreading.jl")
include("optimize.jl")
include("docstrings.jl")

end # module NLLSModel
