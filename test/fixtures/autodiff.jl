using NLLSModel, Test, Static, StaticArrays, Random

# r = A [x; y] - b, for an x of N1 and a y of N2 dimensions, M rows
struct WideLinear{M, N1, N2, L, ML} <: NLLSModel.AbstractResidual
    A::SMatrix{M, L, Float64, ML}
    b::SVector{M, Float64}
end
WideLinear{N1, N2}(A::SMatrix{M, L, Float64, ML}, b) where {M, N1, N2, L, ML} = (@assert L == N1 + N2; WideLinear{M, N1, N2, L, ML}(A, b))
NLLSModel.ndeps(::WideLinear) = static(2)
NLLSModel.nres(::WideLinear{M}) where M = static(M)
NLLSModel.varindices(::WideLinear) = SVector(1, 2)
NLLSModel.getvars(::WideLinear{M, N1, N2}, vars::AbstractVector) where {M, N1, N2} = (vars[1]::SVector{N1, Float64}, vars[2]::SVector{N2, Float64})
NLLSModel.computeresidual(r::WideLinear, x, y) = r.A * vcat(x, y) - r.b
Base.eltype(::WideLinear) = Float64

# The same residual without Base.eltype, and a cost without it (AN8)
struct NoEltype{M} <: NLLSModel.AbstractResidual
    m::M
end
NLLSModel.ndeps(::NoEltype) = static(1)
NLLSModel.nres(r::NoEltype) = r.m
NLLSModel.varindices(::NoEltype) = SVector(1)
NLLSModel.computeresidual(r::NoEltype, x) = x[1:NLLSModel.dynamic(r.m)]
struct NoEltypeCost <: NLLSModel.AbstractCost end
NLLSModel.ndeps(::NoEltypeCost) = static(1)
NLLSModel.varindices(::NoEltypeCost) = SVector(1)
NLLSModel.computecost(::NoEltypeCost, x) = sum(abs2, x)
