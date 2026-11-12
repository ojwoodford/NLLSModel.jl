using Static, SparseArrays
import OptimizationVariableInterface: nvars

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

function Base.getproperty(p::Model, s::Symbol)
    if s === :varnext
        if haskey(task_local_storage(), :nlls_varnext)
            val = task_local_storage(:nlls_varnext)
            if val isa Vector
                return val
            else
                v = copy(getfield(p, :variables))
                unfixed = get(task_local_storage(), :nlls_unfixed, 1)
                v[unfixed] = val
                return v
            end
        end
        return getfield(p, :variables)
    elseif s === :varbest
        if haskey(task_local_storage(), :nlls_varbest)
            val = task_local_storage(:nlls_varbest)
            if val isa Vector
                return val
            else
                v = copy(getfield(p, :variables))
                unfixed = get(task_local_storage(), :nlls_unfixed, 1)
                v[unfixed] = val
                return v
            end
        end
        return getfield(p, :variables)
    end
    return getfield(p, s)
end

function Base.show(io::IO, x::Model)
    print(io, "Model with ", length(x.variables), " variable blocks of total dimension ", UInt(sum(nvars, x.variables; init=0)), 
           ",\n             and ", countcosts(costnum, x.costs), " residual blocks of total dimension ", countcosts(resnum, x.costs), ".")
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::Integer)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if in(unfixed, varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::BitVector)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if any(j -> unfixed[j], varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector{T}, unfixed::Union{Vector{Bool}, SubArray{Bool}}) where T
    selectcosts!(outcosts, incosts, BitVector(unfixed))
end

function selectcosts!(outcosts::CostStruct, incosts::Vector{T}, vec::Vector) where T
    outvec = empty!(get(outcosts, T))
    if !isempty(vec)
        append!(outvec, @inbounds(view(incosts, vec)))
    end
end

# Produce a subproblem containing only the relevant costs
function subproblem(problem::Model{VT, CT}, unfixed) where {VT, CT}
    coststruct = CostStruct{CT}()
    for costs in values(problem.costs)
        selectcosts!(coststruct, costs, unfixed)
    end
    return Model{VT, CT}(problem.variables, coststruct)
end

function selectcosts!(outcosts::CostStruct, incosts::Vector{T}, costindices, lastcost, index) where T
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

function selectcosts!(outcosts::CostStruct, incosts::CostStruct, costindices)
    lastcost = 0
    index = 1
    for costs in values(incosts)
        lastcost, index = selectcosts!(outcosts, costs, costindices, lastcost, index)
    end
end

function subproblem!(to::Model{VT, CT}, from::Model{VT, CT}, costindices::AbstractVector) where {VT, CT}
    selectcosts!(to.costs, from.costs, costindices)
    return to
end

function addcost!(problem::Model, cost::Cost) where Cost <: AbstractCost
    N = ndeps(cost)
    @assert isa(N, StaticInt) && N > 0 && N <= MAX_ARGS "Problem with ndeps()"
    @assert length(varindices(cost)) == N "Problem with varindices()"
    @assert allunique(varindices(cost)) "A cost cannot depend on the same variable more than once"
    vars = getvars(cost, problem.variables)
    @assert isa(vars, Tuple) && isconcretetype(typeof(vars)) && length(vars) == N "getvars(::$(typeof(cost)), vars) should return a concretely typed tuple of length $(dynamic(N)), but returned $(typeof(vars))."
    @assert (!(Cost <: AbstractAdaptiveResidual) || isa(vars[1], AbstractAdaptiveRobustifier)) "Adaptive residual without adaptive robustifier"
    if Cost <: AbstractResidual
        M = nres(cost)
        @assert (isa(M, Integer) || isa(M, StaticInt)) && M > 0 "Problem with nres()"
    end
    push!(problem.costs, cost)
    problem.varcostmapvalid = false
    return nothing
end

function addvariable!(problem::Model, variable)
    N = nvars(variable)
    @assert (isa(N, Integer) || isa(N, StaticInt)) && N > 0 "Problem with nvars()"
    @assert N <= typemax(UInt8) "Variables with more than $(Int(typemax(UInt8))) dimensions are not supported (got $(Int(dynamic(N))))"
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

"""
    partition_schur_problem(problem::Model, unfixed::AbstractVector, linsolver_req=nothing)

Validate bipartite structure, check static eliminable variable dimensions (Dp in {1, 2, 3}),
and perform two-pass variable ordering for Schur complement reduction.
Returns named tuple `(unfixed_inds, extra_blocks, c_dims, p_dims, observations, ncblocks, npblocks)`,
where `extra_blocks` lists the pairs of retained blocks that share a cost (in the sorted retained ordering)
or `nothing` if the problem does not have both retained and eliminable variables, or (unless a Schur solver
was explicitly requested via `linsolver_req`) if its structure does not suit Schur reduction.
"""
function partition_schur_problem(problem::Model, unfixed::AbstractVector, linsolver_req=nothing)
    unfixed_raw = findall(unfixed)
    nblocks = length(unfixed_raw)
    nblocks > 1 || return nothing
    # Only throw if a Schur solver was explicitly requested; otherwise the caller falls back to a non-Schur solver
    explicit_schur = linsolver_req in (:schur, :dense_schur, :sparse_schur)

    # 1. Identify eliminable variables using iseliminable trait
    is_elim = [dynamic(OptimizationVariableInterface.iseliminable(typeof(problem.variables[i]))) for i in unfixed_raw]
    nelim = count(is_elim)
    nret = nblocks - nelim

    if nelim == 0 || nret == 0
        if linsolver_req in (:schur, :dense_schur, :sparse_schur)
            throw(ArgumentError("Schur complement solver requires at least one retained and one eliminable variable (got $nret retained, $nelim eliminable)."))
        end
        return nothing
    end

    # 2. Structural Bipartite Validation
    elim_set = Set(unfixed_raw[findall(is_elim)])
    camera_set = Set(unfixed_raw[findall(!, is_elim)])

    # Validate eliminable variable static dimension in {1, 2, 3}
    block_sizes = [Int(dynamic(nvars(problem.variables[i]))) for i in unfixed_raw]
    for (idx, e) in enumerate(is_elim)
        if e
            d = block_sizes[idx]
            if d < 1 || d > 3
                explicit_schur || return nothing
                throw(ArgumentError("Eliminable variables must have static dimension in {1, 2, 3}, got $d for variable $(unfixed_raw[idx])."))
            end
        end
    end

    # Map from original variable index to camera block index (1:nret)
    cam_raw_list = unfixed_raw[findall(!, is_elim)]
    cam_map = Dict{Int, Int}(var_idx => i for (i, var_idx) in enumerate(cam_raw_list))

    observations_dict = Dict{Int, Vector{Int}}()
    for var_idx in elim_set
        observations_dict[var_idx] = Int[]
    end

    # Pairs of retained variables that share a cost (they need a slot in the reduced system's pattern)
    retained_pairs = Set{Tuple{Int, Int}}()

    # Block-Diagonality Check: No residual connects > 1 unfixed eliminable variable
    for costvec in values(problem.costs)
        for cost in costvec
            vinds = varindices(cost)
            elims_in_cost = [v for v in vinds if v in elim_set]
            if length(elims_in_cost) > 1
                explicit_schur || return nothing
                throw(ArgumentError("Bipartite structure violation: cost block of type $(typeof(cost)) connects multiple eliminable variables $(elims_in_cost). Eliminable variables cannot couple to each other."))
            elseif length(elims_in_cost) == 1
                lm = elims_in_cost[1]
                for v in vinds
                    if v in camera_set
                        push!(observations_dict[lm], cam_map[v])
                    end
                end
            end
            cams_in_cost = [cam_map[v] for v in vinds if v in camera_set]
            for a in 2:length(cams_in_cost), b in 1:a-1
                ca, cb = cams_in_cost[a], cams_in_cost[b]
                push!(retained_pairs, (min(ca, cb), max(ca, cb)))
            end
        end
    end

    # Deduplicate observations per landmark
    observations = Vector{Vector{Int}}(undef, nblocks)
    for (i, var_idx) in enumerate(unfixed_raw)
        if is_elim[i]
            observations[i] = unique(observations_dict[var_idx])
        else
            observations[i] = Int[]
        end
    end

    # 3. Two-pass variable ordering using sort_schur_blocks from BlockLinearSystems
    perm = sort_schur_blocks(is_elim, block_sizes, observations)
    unfixed_inds = unfixed_raw[perm]
    sorted_sizes = block_sizes[perm]

    c_dims = UInt8.(sorted_sizes[1:nret])
    p_dims = UInt8.(sorted_sizes[nret+1:nblocks])

    # Re-map camera indices in observations to new camera ordering
    old_to_new_cam = Dict{Int, Int}()
    for new_c in 1:nret
        orig_var = unfixed_inds[new_c]
        old_c = cam_map[orig_var]
        old_to_new_cam[old_c] = new_c
    end

    obs_remapped = Vector{Vector{Int}}(undef, nelim)
    for l in 1:nelim
        orig_idx = perm[nret + l]
        orig_obs = observations[orig_idx]
        obs_remapped[l] = sort([old_to_new_cam[c] for c in orig_obs if haskey(old_to_new_cam, c)])
    end

    extra_blocks = sort!([(old_to_new_cam[a], old_to_new_cam[b]) for (a, b) in retained_pairs])

    return (
        unfixed_inds = unfixed_inds,
        extra_blocks = extra_blocks,
        c_dims = c_dims,
        p_dims = p_dims,
        observations = obs_remapped,
        ncblocks = nret,
        npblocks = nelim
    )
end
