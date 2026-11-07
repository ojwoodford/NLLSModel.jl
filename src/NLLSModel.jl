module NLLSModel

using BlockMatrices
using FastNorm
using OptimizationVariableInterface
using SIRBSDispatch
using Static, StaticArrays, SparseArrays, LinearAlgebra

import BlockMatrices:
    BlockSparseMatrix, BlockDenseMatrix,
    block, zero!, copy_structure,
    symmetrifyfull, makesparseindices,
    block_sparse_nnz, computestartindices, find_diag_inds

import OptimizationVariableInterface:
    nvars, update, update!, syncvars!,
    ZeroToInfScalar, ZeroToOneScalar

import FastNorm: sqnorm
import SIRBSDispatch: sirbsdispatch

# Export NLLSModel public interface
# Problem construction & evaluation
export Model, NLPLSModel, NLLSNLSModel
export addvariable!, addcost!, addresidual!, subproblem, subproblem!
export cost, optimvars, optimvars_nls, flat_optimvars_nls, nls_callbacks, syncvars!, nresiduals
export isadditive, islinearkernel, clamped_count, reset_clamped_count!

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
include("flatvars.jl")
include("optimize.jl")
include("tall_system.jl")
include("docstrings.jl")

end # module NLLSModel
