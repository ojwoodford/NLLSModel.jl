mutable struct Model{VarTypes, CostTypes, CostVecs, TCT<:Tuple}
    # User provided
    costs::CostStruct{TCT, CostVecs} # Set of cost and residual blocks that define the problem
    variables::Vector{VarTypes}  # Current state of variables in use by optimizer

    # Used internally
    varcostmap::SparseMatrixCSC{Bool, Int} # Map of variables used by each cost block
    varcostmapvalid::Bool        # Flag indicating whether the above variable cost map is valid

    # Constructors
    function Model{VarTypes, CostTypes}(
        vars::Vector{VarTypes} = Vector{VarTypes}(),
        costs::CostStruct = CostStruct{CostTypes}(),
        varcostmap = spzeros(Bool, 0, 0),
        varcostmapvalid = false
    ) where {VarTypes, CostTypes}
        TCT = tupelize(CostTypes)
        costs isa CostStruct{TCT} || throw(ArgumentError("Expected costs of type CostStruct{$TCT}, got $(typeof(costs))"))
        return new{VarTypes, CostTypes, typeof(costs.data), TCT}(costs, vars, varcostmap, varcostmapvalid)
    end
    function Model{VarTypes, CostTypes, CostVecs, TCT}(
        vars::Vector{VarTypes},
        costs::CostStruct{TCT, CostVecs},
        varcostmap = spzeros(Bool, 0, 0),
        varcostmapvalid = false
    ) where {VarTypes, CostTypes, CostVecs, TCT}
        return new{VarTypes, CostTypes, CostVecs, TCT}(costs, vars, varcostmap, varcostmapvalid)
    end
end

Model(::Type{VT}, ::Type{CT}) where {VT, CT} = Model{VT, CT}()
Model(v::Vector{VT}, ::Type{CT}) where {VT, CT} = Model{VT, CT}(v)
Model(v::Vector{VT}, r::CostStruct{CT}) where {VT, CT} = Model{VT, CT}(v, r)

function Base.show(io::IO, x::Model)
    print(io, "Model with ", length(x.variables), " variable blocks of total dimension ", UInt(sum(nvars, x.variables; init=0)), 
           ",\n             and ", countcosts(costnum, x.costs), " residual blocks of total dimension ", countcosts(resnum, x.costs), ".")
end

# The costs of `incosts` that touch an unfixed variable (`unfixed` as `convertunfixed` returns it: an index or a mask)
function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::Integer)
    selectcostindices!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if in(unfixed, varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::BitVector)
    selectcostindices!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if any(j -> unfixed[j], varindices(r))])
end

# The costs of `incosts` at the cost indices `vec`
function selectcostindices!(outcosts::CostStruct, incosts::Vector{T}, vec::Vector) where T
    outvec = empty!(get(outcosts, T))
    if !isempty(vec)
        append!(outvec, @inbounds(view(incosts, vec)))
    end
end

# Produce a subproblem containing only the relevant costs
function subproblem(problem::Model{VT, CT}, unfixed) where {VT, CT}
    unfixed = convertunfixed(unfixed, problem)
    coststruct = CostStruct{CT}()
    for costs in values(problem.costs)
        selectcosts!(coststruct, costs, unfixed)
    end
    return Model{VT, CT}(problem.variables, coststruct)
end

# The costs at the sorted, in-range indices `costindices` into all costs of `incosts`, in order (starting from `index`)
function selectcostindices!(outcosts::CostStruct, incosts::Vector{T}, costindices, lastcost, index) where T
    tocostvec = get!(outcosts, T)::Vector{T}
    empty!(tocostvec)
    firstind = lastcost
    lastcost += length(incosts)
    len = length(costindices)
    while index <= len && @inbounds(costindices[index]) <= lastcost
        push!(tocostvec, @inbounds(incosts[costindices[index]-firstind]))
        index += 1
    end
    return lastcost, index
end

function selectcostindices!(outcosts::CostStruct, incosts::CostStruct, costindices)
    lastcost = 0
    index = 1
    for costs in values(incosts)
        lastcost, index = selectcostindices!(outcosts, costs, costindices, lastcost, index)
    end
end

function subproblem!(to::Model{VT, CT}, from::Model{VT, CT}, costindices::AbstractVector) where {VT, CT}
    issorted(costindices) || throw(ArgumentError("subproblem!: the cost indices must be sorted in ascending order"))
    ncosts = countcosts(costnum, from.costs)
    isempty(costindices) || (first(costindices) >= 1 && last(costindices) <= ncosts) ||
        throw(ArgumentError("subproblem!: the cost indices must be between 1 and the number of costs, $ncosts"))
    selectcostindices!(to.costs, from.costs, costindices)
    return to
end

# A user's cost or variable that breaks a rule of addcost! or addvariable!
inputerror(msg, ::Type{T}, what) where T = throw(ArgumentError("$msg ($what of type $T)"))

function addcost!(problem::Model, cost::Cost) where Cost <: AbstractCost
    N = ndeps(cost)
    isa(N, StaticInt) && N > 0 && N <= MAX_ARGS || inputerror("Problem with ndeps()", Cost, "cost")
    length(varindices(cost)) == N || inputerror("Problem with varindices()", Cost, "cost")
    allunique(varindices(cost)) || inputerror("A cost cannot depend on the same variable more than once", Cost, "cost")
    vars = getvars(cost, problem.variables)
    isa(vars, Tuple) && isconcretetype(typeof(vars)) && length(vars) == N ||
        throw(ArgumentError("getvars(::$(typeof(cost)), vars) should return a concretely typed tuple of length $(dynamic(N)), but returned $(typeof(vars))."))
    !(Cost <: AbstractAdaptiveResidual) || isa(vars[1], AbstractAdaptiveRobustifier) || inputerror("Adaptive residual without adaptive robustifier", Cost, "cost")
    if Cost <: AbstractResidual
        M = nres(cost)
        (isa(M, Integer) || isa(M, StaticInt)) && M > 0 || inputerror("Problem with nres()", Cost, "cost")
    end
    push!(problem.costs, cost)
    problem.varcostmapvalid = false
    return nothing
end

function addvariable!(problem::Model, variable)
    N = nvars(variable)
    (isa(N, Integer) || isa(N, StaticInt)) && N > 0 || inputerror("Problem with nvars()", typeof(variable), "variable")
    push!(problem.variables, variable)
    problem.varcostmapvalid = false
    return length(problem.variables)
end

function updatevarcostmap!(varcostmap::SparseMatrixCSC{Bool, Int}, costvec::Vector, colind::Int, rowind::Int)
    numres = length(costvec)
    if numres > 0
        @inbounds for cost in costvec
            ndeps_ = dynamic(ndeps(cost))
            if ndeps_ == 1
                varcostmap.rowval[rowind] = varindices(cost)[1]
            else
                varcostmap.rowval[SR(0, ndeps_-1).+rowind] .= sort(varindices(cost))
            end
            rowind += ndeps_
            colind += 1
            varcostmap.colptr[colind] = rowind
        end
    end
    return colind, rowind
end

function updatevarcostmap!(varcostmap::SparseMatrixCSC{Bool, Int}, costs::CostStruct)
    resize!(varcostmap.rowval, countcosts(costdeps, costs))
    prevlen = length(varcostmap.nzval)
    resize!(varcostmap.nzval, length(varcostmap.rowval))

    varcostmap.nzval[prevlen+1:length(varcostmap.rowval)] .= true
    varcostmap.colptr[1] = 1
    colind = 1
    rowind = 1
    @inbounds for costvec in values(costs)
        colind, rowind = updatevarcostmap!(varcostmap, costvec, colind, rowind)
    end
    return nothing
end

function updatevarcostmap!(problem::Model)
    sz = (length(problem.variables), countcosts(costnum, problem.costs))
    if size(problem.varcostmap) != sz
        problem.varcostmap = spzeros(Bool, sz[1], sz[2])
    end
    retval = updatevarcostmap!(problem.varcostmap, problem.costs)
    problem.varcostmapvalid = true
    return retval
end

function getvarcostmap(problem::Model)
    if !problem.varcostmapvalid
        updatevarcostmap!(problem)
    end
    return problem.varcostmap
end

costnum(vec::Vector)  = length(vec)
costdeps(vec::Vector) = length(vec) != 0 ? length(vec) * ndeps(vec[1]) : 0
resnum(vec::Vector)   = length(vec) != 0 ? (dynamic(is_static(nres(vec[1]))) ? length(vec) * nres(vec[1]) : sum(nres, vec; init=0)) : 0
countcosts(fun, costs::CostStruct) = sum(fun, values(costs); init=0)
nresiduals(problem::Model) = Int(countcosts(resnum, problem.costs))

# The partition for elimination, or `nothing`: the one rule of the Hessian and the Jacobian path. `:none` never eliminates;
# `:auto` eliminates when the model has eliminable variables and the structure fits; `:eliminate` eliminates or throws.
function schur_partition(problem::Model, unfixed::AbstractVector, reduction::Symbol)
    reduction === :none && return nothing
    has_elims = any(v -> dynamic(OptimizationVariableInterface.iseliminable(typeof(v))), problem.variables)
    (reduction === :eliminate || has_elims) || return nothing
    return partition_schur_problem(problem, unfixed, reduction === :eliminate)
end

# The row blocks of a description held as one flat vector of block numbers and a pointer vector: row block `k` is a view,
# so no vector is allocated per cost
struct FlatRowBlocks <: AbstractVector{SubArray{Int, 1, Vector{Int}, Tuple{UnitRange{Int}}, true}}
    flat::Vector{Int}
    ptr::Vector{Int}
end
Base.size(r::FlatRowBlocks) = (length(r.ptr) - 1,)
Base.@propagate_inbounds Base.getindex(r::FlatRowBlocks, k::Int) = view(r.flat, r.ptr[k]:r.ptr[k + 1] - 1)

"""
    partition_schur_problem(problem::Model, unfixed::AbstractVector, explicit::Bool=false)

Decide whether the problem is eliminated, and which variables: the unfixed eliminable ones (`iseliminable`). Returns the
description BlockLinearSystems lays the elimination out from, in the order of `findall(unfixed)`:
`(unfixed_raw, blocksizes, eliminated, rowblocks)`, with one row block per cost (in `eachcost` order) listing the cost's
unfixed variables by their position in `unfixed_raw`. Returns `nothing` if the problem does not have both retained and
eliminable variables, or if a cost connects two eliminable variables. With `explicit = true` (`reduction = :eliminate` in
[`optimvars`](@ref)) a structure that does not fit throws an `ArgumentError` instead. Eliminated variables, like retained
ones, may have any number of dimensions that BlockLinearSystems takes as a block; it checks the sizes, and its
`ArgumentError` names the block.
"""
function partition_schur_problem(problem::Model, unfixed::AbstractVector, explicit::Bool=false)
    unfixed_raw = findall(unfixed)
    nblocks = length(unfixed_raw)
    nblocks > 1 || return nothing
    # Only throw if elimination was explicitly requested; otherwise the caller falls back to no elimination

    # 1. Identify eliminable variables using iseliminable trait
    is_elim = [dynamic(OptimizationVariableInterface.iseliminable(typeof(problem.variables[i]))) for i in unfixed_raw]
    nelim = count(is_elim)
    nret = nblocks - nelim

    if nelim == 0 || nret == 0
        if explicit
            throw(ArgumentError("Schur complement solver requires at least one retained and one eliminable variable (got $nret retained, $nelim eliminable)."))
        end
        return nothing
    end

    # 2. The row blocks, and the check that no cost connects more than one unfixed eliminable variable
    blockof = zeros(Int, length(problem.variables))   # Position of each unfixed variable in unfixed_raw, 0 if fixed
    for (b, v) in enumerate(unfixed_raw)
        blockof[v] = b
    end
    ptr = Int[1]
    flat = Int[]
    for costvec in values(problem.costs)
        for cost in costvec
            vinds = varindices(cost)
            nelims = 0
            for v in vinds
                b = blockof[v]
                b == 0 && continue
                nelims += is_elim[b]
                push!(flat, b)
            end
            if nelims > 1
                explicit || return nothing
                elims_in_cost = [v for v in vinds if blockof[v] != 0 && is_elim[blockof[v]]]
                throw(ArgumentError("Bipartite structure violation: cost block of type $(typeof(cost)) connects multiple eliminable variables $(elims_in_cost). Eliminable variables cannot couple to each other."))
            end
            push!(ptr, length(flat) + 1)
        end
    end

    # Blocks of any size BlockLinearSystems takes (it checks them), eliminated or retained
    blocksizes = [block_dimension(problem.variables[i]) for i in unfixed_raw]
    return (unfixed_raw = unfixed_raw, blocksizes = blocksizes, eliminated = is_elim, rowblocks = FlatRowBlocks(flat, ptr))
end
