using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra

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

# AL1: the dense `fdf!` fills whichever Hessian matrix it is handed. The optimizer alternates two, the current point's
# and the trial's: evaluate at `x` into one and at a perturbed point into the other, twice over, and each matrix (filled
# with NaN first, so every entry must be written) holds its own point's cost, gradient and Hessian, as a single matrix
# evaluated at that point does, and is not changed by the evaluation into the other.
function alternating_dense_ok(fdf!, x)
    m = Int(sum(v -> Int(NLLSModel.nvars(v)), x))
    xp = [v .+ 0.1 for v in x]
    g0, H0, gp0, Hp0 = zeros(m), zeros(m, m), zeros(m), zeros(m, m)
    c0, cp0 = fdf!(g0, H0, x), fdf!(gp0, Hp0, xp)
    g1, H1, g2, H2 = zeros(m), fill(NaN, m, m), zeros(m), fill(NaN, m, m)
    ok = c0 != cp0 && H0 != Hp0 # the two points differ
    for _ in 1:2
        ok &= fdf!(g1, H1, x) == c0 && g1 == g0 && H1 == H0
        ok &= fdf!(g2, H2, xp) == cp0 && g2 == gp0 && H2 == Hp0
        ok &= H1 == H0
    end
    return ok
end
