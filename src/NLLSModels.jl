module NLLSModels

using BlockMatrices
using FastLinAlg
using OptimizationVariableInterface
using ValueDispatch
using Static, StaticArrays, SparseArrays, LinearAlgebra

import BlockMatrices:
    BlockSparseMatrix, BlockDenseMatrix,
    block, validblock, zero!, copy_structure, uniformscaling!,
    symmetrifyfull, symmetrifyfull!, symmetrifysparse,
    makesparseindices, cacheindices, clearindicescache, updatelastrowsym!,
    sparse_dense_decision, block_sparse_nnz, computestartindices, find_diag_inds

import OptimizationVariableInterface:
    nvars, update, update!, copyvars, syncvars!, isbitsvartype,
    EuclideanVector, DynamicVector, ZeroToInfScalar, ZeroToOneScalar

import FastLinAlg: sqnorm, fast_dot, fast_norm, fast_bAb
import ValueDispatch: valuedispatch

# Export NLLSModels types and functions
export NLLSProblem
export AbstractCost, AbstractResidual, AbstractAdaptiveResidual, AbstractRobustifier, AbstractAdaptiveRobustifier
export NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel, ContaminatedGaussian
export EuclideanVector, DynamicVector, ZeroToInfScalar, ZeroToOneScalar
export SimpleError2, SimpleError3, SimpleError4
export BlockSparseMatrix, BlockDenseMatrix, VectorRepo
export addvariable!, addcost!, addresidual!, updatevarcostmap!, subproblem, subproblem!
export cost, costgradhess!, computeresidual, computeresjac, computecost, computecostgradhess
export optimvars
export getvars, varindices, generatemeasurement, ndeps, nres
export robustkernel, robustify, robustifydcost, robustifydkernel
export nvars, update, update!, copyvars, syncvars!, isbitsvartype, zero!

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
include("VectorRepo.jl")
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

end # module NLLSModels
