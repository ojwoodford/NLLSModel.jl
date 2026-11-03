using Static, SparseArrays
import OptimizationVariableInterface: nvars

const CostStruct = VectorRepo

mutable struct NLLSProblem{VarTypes, CostTypes}
    # User provided
    costs::CostStruct{CostTypes} # Set of cost and residual blocks that define the problem
    variables::Vector{VarTypes}  # Current state of variables in use by optimizer

    # Used internally
    varcostmap::SparseMatrixCSC{Bool, Int} # Map of variables used by each cost block
    varcostmapvalid::Bool        # Flag indicating whether the above variable cost map is valid

    # Constructor
    function NLLSProblem{VarTypes, CostTypes}(
        vars = Vector{VarTypes}(),
        costs = CostStruct{CostTypes}(),
        varcostmap = spzeros(Bool, 0, 0),
        varcostmapvalid = false
    ) where {VarTypes, CostTypes}
        return new{VarTypes, CostTypes}(costs, vars, varcostmap, varcostmapvalid)
    end
end

NLLSProblem() = NLLSProblem{Any, Any}()
NLLSProblem(VT::Union{DataType, Union}) = NLLSProblem{VT, Any}()
NLLSProblem(VT::Union{DataType, Union}, CT::Union{DataType, Union}) = NLLSProblem{VT, CT}()
NLLSProblem(v::Vector{VT}) where VT = NLLSProblem{VT, Any}(v)
NLLSProblem(v::Vector{VT}, r::CostStruct{CT}) where {VT, CT} = NLLSProblem{VT, CT}(v, r)

function Base.getproperty(p::NLLSProblem, s::Symbol)
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

function Base.show(io::IO, x::NLLSProblem)
    print(io, "NLLSProblem with ", length(x.variables), " variable blocks of total dimension ", UInt(sum(nvars, x.variables)), 
           ",\n             and ", countcosts(costnum, x.costs), " residual blocks of total dimension ", countcosts(resnum, x.costs), ".")
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::Integer)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if in(unfixed, varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::BitVector)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if any(j -> unfixed[j], varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector{T}, vec::Vector) where T
    if !isempty(vec)
        outcosts.data[T] = @inbounds incosts[vec]
    end
end

# Produce a subproblem containing only the relevant costs
function subproblem(problem::NLLSProblem{VT, CT}, unfixed) where {VT, CT}
    coststruct = CostStruct{CT}()
    for costs in values(problem.costs)
        selectcosts!(coststruct, costs, unfixed)
    end
    return NLLSProblem{VT, CT}(problem.variables, coststruct)
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

function subproblem!(to::NLLSProblem{VT, CT}, from::NLLSProblem{VT, CT}, costindices::AbstractVector) where {VT, CT}
    selectcosts!(to.costs, from.costs, costindices)
    return to
end

function addcost!(problem::NLLSProblem, cost::Cost) where Cost <: AbstractCost
    N = ndeps(cost)
    @assert isa(N, StaticInt) && N > 0 && N <= MAX_ARGS "Problem with ndeps()"
    @assert length(varindices(cost)) == N "Problem with varindices()"
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

function addvariable!(problem::NLLSProblem, variable)
    N = nvars(variable)
    @assert (isa(N, Integer) || isa(N, StaticInt)) && N > 0 "Problem with nvars()"
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

function updatevarcostmap!(problem::NLLSProblem)
    sz = (length(problem.variables), countcosts(costnum, problem.costs))
    if size(problem.varcostmap) != sz
        problem.varcostmap = spzeros(Bool, sz[1], sz[2])
    end
    retval = updatevarcostmap!(problem.varcostmap, problem.costs)
    problem.varcostmapvalid = true
    return retval
end

function getvarcostmap(problem::NLLSProblem)
    if !problem.varcostmapvalid
        updatevarcostmap!(problem)
    end
    return problem.varcostmap
end

function reordercostsforschur!(problem::NLLSProblem, schurvars)
    schurruns = Dict{DataType, Vector{Int}}()
    costvarmap = sparse(getvarcostmap(problem)')[:,schurvars]
    schurvarpercost = sum(costvarmap; dims=2)
    @assert all(x->(x<=1), schurvarpercost) "Each cost block can only depend on one schur variable at most"
    schurvarpercost[schurvarpercost .> 0] = costvarmap.rowval
    startind = 0
    for costs in values(problem.costs)
        if !isempty(costs)
            schurvarpercostsubset = view(schurvarpercost, startind+1:startind+length(costs))
            startind += length(costs)
            sortedorder = sortperm(schurvarpercostsubset)
            permute!(schurvarpercostsubset, sortedorder)
            schurruns[typeof(costs[1])] = runlengthencodesortedints(schurvarpercostsubset)
            permute!(costs, sortedorder)
        end
    end
    problem.varcostmapvalid = false
    return schurruns
end

costnum(vec::Vector)  = length(vec)
costdeps(vec::Vector) = length(vec) != 0 ? length(vec) * ndeps(vec[1]) : 0
resnum(vec::Vector)   = length(vec) != 0 ? (dynamic(is_static(nres(vec[1]))) ? length(vec) * nres(vec[1]) : sum(nres, vec; init=0)) : 0
countcosts(fun, costs::CostStruct{Any}) = sum(fun, values(costs); init=0)
countcosts(fun, costs::CostStruct{T}) where T = countcosts(fun, costs, T)
countcosts(fun, costs, T::Union) = countcosts(fun, costs, T.a) + countcosts(fun, costs, T.b)
countcosts(fun, costs, T::DataType) = fun(get(costs, T))
