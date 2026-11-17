# Chain residual between neighbouring 2-vectors, with static (SVector) or run-time (Vector) sized variables
struct ChainRes{V} <: NLLSModel.AbstractResidual
    meas::SVector{2, Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::ChainRes) = static(2)
NLLSModel.nres(::ChainRes) = static(2)
NLLSModel.varindices(r::ChainRes) = SVector(r.i, r.j)
NLLSModel.getvars(r::ChainRes{V}, vars::AbstractVector) where V = (vars[r.i]::V, vars[r.j]::V)
NLLSModel.computeresidual(r::ChainRes, a, b) = SVector(a[1] - b[1] + 0.1 * a[2] * b[2], a[2] - b[2]) - r.meas
Base.eltype(::ChainRes) = Float64

function chainproblem(::Type{V}, n) where V
    problem = Model(V, ChainRes{V})
    for i in 1:n
        addvariable!(problem, V([sin(i), cos(2i)]))
    end
    for i in 1:n-1
        addcost!(problem, ChainRes{V}(SVector(0.1i, -0.2), i, i + 1))
    end
    return problem
end

# Residuals of length one may return a number or a one-element vector (M5)
struct OneRes{S} <: NLLSModel.AbstractResidual
    meas::Float64
end
NLLSModel.ndeps(::OneRes) = static(1)
NLLSModel.nres(::OneRes) = static(1)
NLLSModel.varindices(::OneRes) = 1
NLLSModel.getvars(::OneRes, vars::Vector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(r::OneRes{true}, x) = x[1] * x[2] - r.meas
NLLSModel.computeresidual(r::OneRes{false}, x) = SVector(x[1] * x[2] - r.meas)
Base.eltype(::OneRes) = Float64
