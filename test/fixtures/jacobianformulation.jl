using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization
import OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars

fixture("jacobianproblems")

const JacobianExt = Base.get_extension(NLLSModel, :SecondOrderOptimizationExt)
const JacobianOptions = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200)

# Records whether any residual was evaluated off the calling thread, to see whether an evaluation was threaded
const SPY_CALLER = Ref(0)
const SPY_OFFTHREAD = Threads.Atomic{Int}(0)
struct SpyRes <: NLLSModel.AbstractResidual
    t::Float64
end
NLLSModel.ndeps(::SpyRes) = static(1)
NLLSModel.nres(::SpyRes) = static(1)
NLLSModel.varindices(::SpyRes) = 1
NLLSModel.getvars(::SpyRes, vars::AbstractVector) = (vars[1]::Float64,)
function NLLSModel.computeresidual(r::SpyRes, x)
    Threads.threadid() == SPY_CALLER[] || Threads.atomic_add!(SPY_OFFTHREAD, 1)
    return SVector(x - r.t)
end
Base.eltype(::SpyRes) = Float64
function spymodel(ncosts, nvars=1)
    model = Model(Float64, SpyRes)
    for _ in 1:nvars
        addvariable!(model, 0.0)
    end
    for k in 1:ncosts
        addcost!(model, SpyRes(sin(k)))
    end
    return model
end
# Whether `fun()` evaluated a residual off the calling thread
function offthread(fun)
    SPY_CALLER[] = Threads.threadid()
    SPY_OFFTHREAD[] = 0
    fun()
    return SPY_OFFTHREAD[] > 0
end

# Compare the Jacobian solve with the Hessian formulation: same final cost (the gradient at an accurate minimum is rounding noise, so it
# is not compared), and the variables to `atol`. `cost0` is the cost at the start, to give the tolerance for zero-residual problems.
function sameoptimum(model, options=JacobianOptions; atol=1e-5, kwargs...)
    c0 = NLLSModel.cost(model)
    solved_hes, rn = optimize(model, options; numthreads=static(1), kwargs...)
    solved_jac, rt = optimize(model, options; formulation=:jacobian, numthreads=static(1), kwargs...)
    cn, ct = NLLSModel.cost(solved_hes), NLLSModel.cost(solved_jac)
    ok = isapprox(ct, cn; rtol=1e-8, atol=1e-10 * c0) && isapprox(ct, rt.bestcost; rtol=1e-12) && samevars(solved_jac, solved_hes; atol) && rt.bestcost < c0
    return ok, ct, cn
end
