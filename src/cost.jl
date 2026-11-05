using Static
import ForwardDiff

cost(problem::NLLSProblem) = cost(problem.variables, problem.costs)
cost(problem::NLLSProblem, numthreads::Integer) = cost(problem.variables, problem.costs, static(numthreads))
cost(problem::NLLSProblem, numthreads::StaticInt) = cost(problem.variables, problem.costs, numthreads)
cost(problem::NLLSProblem, options) = cost(problem.variables, problem.costs, options.numthreads)
cost(vars::AbstractVector, costs::CostStruct)::Float64 = cost(vars, costs, static(1))
cost(vars::AbstractVector, costs::CostStruct, ::StaticInt{1})::Float64 = cost_struct(vars, costs, static(1))
cost(vars::AbstractVector, costs::CostStruct, numthreads::Integer)::Float64 = cost(vars, costs, static(numthreads))
cost(vars::AbstractVector, costs::CostStruct, numthreads::StaticInt)::Float64 = cost_struct(vars, costs, numthreads)

cost_struct(vars, costs::CostStruct, ::StaticInt{1}) = sum(bindleadingargs(computecost, vars), costs; init=0.0)
cost_struct(vars, costs::CostStruct, numthreads::StaticInt) = cost_multithreaded(vars, costs, numthreads)

function cost_range(vars, costlist::AbstractVector, rng::AbstractUnitRange)::Float64
    tot = 0.0
    @inbounds for i in rng
        tot += computecost(vars, costlist[i])
    end
    return tot
end

@inline computecost(vars::AbstractVector, cost::AbstractCost)::Float64 = computecost(cost, getvars(cost, vars)...)

function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::StaticInt)::Float64
    # Compute the residual
    c, g, H = computecostgradhess(varflags, cost, vars...)
    
    # Update the blocks in the problem
    updatesymlinearsystem!(linsystem, g, H, vars, varflags, blockind)

    # Return the cost
    return c
end
function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::StaticInt{0})::Float64
    @warn "A cost for which all variables are fixed is being optimized. Remove such cost functions from the cost to avoid redundant computation, using the subproblem() method."
    return computecost(cost, vars...)
end

function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::Integer)::Float64
    # Common case - all unfixed
    nvars = ndeps(cost)
    maxflags = static(2 ^ dynamic(nvars) - 1)
    if varflags == maxflags
        return gradhesshelper!(linsystem, cost, vars, blockind, maxflags)
    end

    if nvars <= 5
        # Value dispatch gradient computation based on the varflags
        return valuedispatch(static(0), maxflags-static(1), varflags, bindleadingargs(gradhesshelper!, linsystem, cost, vars, blockind))
    else
        # Fall back on dynamic dispatch
        return gradhesshelper!(linsystem, cost, vars, blockind, static(varflags))
    end
end

# Compute the variable flags indicating which variables are unfixed (i.e. to be optimized)
computevarflags(blockind) = mapreduce((ind, val) -> (val != 0) << ind, |, SR(0, length(blockind)-1), blockind)

function costgradhess!(linsystem, vars::AbstractVector, cost::AbstractCost)
    # Determine which variables are free
    blockind = getoffsets(cost, linsystem)
    varflags = computevarflags(blockind)
    # Complete the rest of the computation based on the varflags
    return gradhesshelper!(linsystem, cost, getvars(cost, vars), blockind, varflags)
end

function costgradhess_nozero!(linsystem, vars::AbstractVector, costs::CostStruct)::Float64
    return sum(bindleadingargs(costgradhess!, linsystem, vars), costs; init=0.0)
end

function costgradhess!(linsystem, vars::AbstractVector, costs::CostStruct)::Float64
    zero!(linsystem)
    return costgradhess_nozero!(linsystem, vars, costs)
end
