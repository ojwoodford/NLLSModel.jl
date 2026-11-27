# Precompile workload (improvements.md item 3): optimize! of small fixed models, on the Hessian and the Jacobian path, with
# dense, sparse and eliminated storage, one thread. The residuals and the eliminable variable are this file's own, so what
# is compiled for a user's types is only the part not specialized on them.
struct PrecompilePrior <: NLLSModel.AbstractResidual
    target::SVector{3, Float64}
    v::Int
end
NLLSModel.ndeps(::PrecompilePrior) = static(1)
NLLSModel.nres(::PrecompilePrior) = static(3)
NLLSModel.varindices(r::PrecompilePrior) = SVector(r.v)
NLLSModel.getvars(r::PrecompilePrior, vars::Vector) = (vars[r.v]::SVector{3, Float64},)
NLLSModel.computeresidual(r::PrecompilePrior, x) = x - r.target
Base.eltype(::PrecompilePrior) = Float64

struct PrecompileLink <: NLLSModel.AbstractResidual
    d::Float64
    a::Int
    b::Int
end
NLLSModel.ndeps(::PrecompileLink) = static(2)
NLLSModel.nres(::PrecompileLink) = static(1)
NLLSModel.varindices(r::PrecompileLink) = SVector(r.a, r.b)
NLLSModel.getvars(r::PrecompileLink, vars::Vector) = (vars[r.a]::SVector{3, Float64}, vars[r.b]::SVector{3, Float64})
NLLSModel.computeresidual(r::PrecompileLink, x, y) = SVector(sqrt(sum(abs2, x - y)) - r.d)
Base.eltype(::PrecompileLink) = Float64

struct PrecompilePoint{T}
    p::SVector{3, T}
end
OptimizationVariableInterface.nvars(::PrecompilePoint) = static(3)
OptimizationVariableInterface.update(v::PrecompilePoint, step, offset = 1) = PrecompilePoint(v.p + SVector(step[offset], step[offset + 1], step[offset + 2]))
OptimizationVariableInterface.iseliminable(::Type{<:PrecompilePoint}) = static(true)

struct PrecompileObs <: NLLSModel.AbstractResidual
    z::SVector{2, Float64}
    c::Int
    p::Int
end
NLLSModel.ndeps(::PrecompileObs) = static(2)
NLLSModel.nres(::PrecompileObs) = static(2)
NLLSModel.varindices(r::PrecompileObs) = SVector(r.c, r.p)
NLLSModel.getvars(r::PrecompileObs, vars::Vector) = (vars[r.c]::SVector{6, Float64}, vars[r.p]::PrecompilePoint{Float64})
function NLLSModel.computeresidual(r::PrecompileObs, c, p)
    q = p.p + SVector(c[4], c[5], c[6]) + SVector(c[2] * p.p[3] - c[3] * p.p[2], c[3] * p.p[1] - c[1] * p.p[3], c[1] * p.p[2] - c[2] * p.p[1])
    return SVector(q[1] / q[3], q[2] / q[3]) - r.z
end
Base.eltype(::PrecompileObs) = Float64

@setup_workload begin
    truth(i) = SVector(cos(0.8i), sin(1.1i), 0.3i)
    function precompile_chain(n)
        m = Model(SVector{3, Float64}, Union{PrecompilePrior, PrecompileLink})
        for i in 1:n
            addvariable!(m, truth(i) + SVector(0.05, -0.03, 0.04))
        end
        addcost!(m, PrecompilePrior(truth(1), 1))
        for i in 1:n-1
            addcost!(m, PrecompileLink(sqrt(sum(abs2, truth(i) - truth(i + 1))), i, i + 1))
            addcost!(m, PrecompilePrior(truth(i + 1), i + 1))
        end
        return m
    end
    function precompile_ba()
        m = Model(Union{SVector{6, Float64}, PrecompilePoint{Float64}}, PrecompileObs)
        for i in 1:2
            addvariable!(m, SVector(0.0, 0.0, 0.0, 0.3 * i, 0.0, 0.0))
        end
        for j in 1:6
            addvariable!(m, PrecompilePoint{Float64}(SVector(sin(j), cos(j), 5.0 + 0.1j)))
        end
        for i in 1:2, j in 1:6
            p = SVector(sin(j), cos(j), 5.0 + 0.1j) + SVector(0.3 * i, 0.0, 0.0)
            addcost!(m, PrecompileObs(SVector(p[1] / p[3], p[2] / p[3]) .+ 0.001, i, 2 + j))
        end
        return m
    end
    opts = OptimizationOptions(maxiters = 3)
    @compile_workload begin
        for formulation in (:hessian, :jacobian)
            optimize!(precompile_chain(1), opts; formulation, numthreads = static(1))
            optimize!(precompile_chain(4), opts; formulation, numthreads = static(1), storage = :dense)
            optimize!(precompile_chain(4), opts; formulation, numthreads = static(1), storage = :sparse)
            optimize!(precompile_ba(), opts; formulation, numthreads = static(1), reduction = :auto, storage = :dense)
            optimize!(precompile_ba(), opts; formulation, numthreads = static(1), reduction = :auto, storage = :sparse)
        end
    end
end
