cost(problem::Model) = cost(problem.variables, problem.costs)
cost(problem::Model, numthreads::Integer) = cost(problem.variables, problem.costs, static(numthreads))
cost(problem::Model, numthreads::StaticInt) = cost(problem.variables, problem.costs, numthreads)
cost(vars::AbstractVector, costs::CostStruct)::Float64 = cost(vars, costs, static(1))
cost(vars::AbstractVector, costs::CostStruct, ::StaticInt{1})::Float64 = cost_struct(vars, costs, static(1))
cost(vars::AbstractVector, costs::CostStruct, numthreads::Integer)::Float64 = cost(vars, costs, static(numthreads))
cost(vars::AbstractVector, costs::CostStruct, numthreads::StaticInt)::Float64 = cost_struct(vars, costs, numthreads)

cost_struct(vars, costs::CostStruct, ::StaticInt{1}) = sum(bindleadingargs(computecost, vars), costs; init=0.0)
cost_struct(vars, costs::CostStruct, numthreads::StaticInt) =
    cost_multithreaded(vars, costs, threadcount(costestimate(costs), MIN_WORK_PER_THREAD_COST, dynamic(numthreads), costsplit(costs)))

function cost_range(vars, costlist::AbstractVector, rng::AbstractUnitRange)::Float64
    tot = 0.0
    @inbounds for i in rng
        tot += computecost(vars, costlist[i])
    end
    return tot
end

@inline computecost(vars::AbstractVector, cost::AbstractCost)::Float64 = computecost(cost, getvars(cost, vars)...)

function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::StaticInt)::Float64
    # Compute the cost, gradient and Hessian, and add them to the system. The result is handed over whole (a function
    # barrier): where it is not inferred (run-time sizes), a union of possible result types split here costs a slow
    # dispatch per cost, while one dispatch on the concrete tuple is cached.
    return accumulate_costgradhess!(linsystem, computecostgradhess(varflags, cost, vars...), vars, varflags, blockind)
end
@inline function accumulate_costgradhess!(linsystem, (c, g, H), vars, varflags, blockind)
    updatesymlinearsystem!(linsystem, g, H, vars, varflags, blockind)
    return c
end
function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::StaticInt{0})::Float64
    @warn "A cost for which all variables are fixed is being optimized. Remove such cost functions from the cost to avoid redundant computation, using the subproblem() method." maxlog=1
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
        return sirbsdispatch(static(0), maxflags-static(1), varflags, bindleadingargs(gradhesshelper!, linsystem, cost, vars, blockind))
    else
        # Fall back on dynamic dispatch
        return gradhesshelper!(linsystem, cost, vars, blockind, static(varflags))
    end
end

# Compute the variable flags indicating which variables are unfixed (i.e. to be optimized)
computevarflags(blockind) = mapreduce((ind, val) -> (val != 0) << ind, |, SR(0, length(blockind)-1), blockind)

# One cost into the system. A single-variable system may hold only its lower triangle until `finish!` (called once per
# evaluation by `costgradhess_nozero!`), so anything that reads the system after adding single costs calls it first.
function costgradhess!(linsystem, vars::AbstractVector, cost::AbstractCost)
    # Determine which variables are free
    blockind = getoffsets(cost, linsystem)
    varflags = computevarflags(blockind)
    # Complete the rest of the computation based on the varflags
    return gradhesshelper!(linsystem, cost, getvars(cost, vars), blockind, varflags)
end

function costgradhess_nozero!(linsystem, vars::AbstractVector, costs::CostStruct)::Float64
    c = sum(bindleadingargs(costgradhess!, linsystem, vars), costs; init=0.0)
    finish!(linsystem)
    return c
end

function costgradhess!(linsystem, vars::AbstractVector, costs::CostStruct)::Float64
    zero!(linsystem)
    return costgradhess_nozero!(linsystem, vars, costs)
end
