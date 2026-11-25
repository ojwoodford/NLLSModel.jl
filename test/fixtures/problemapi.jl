using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra

struct DiffResidual <: NLLSModel.AbstractResidual
    i::Int
    j::Int
end
NLLSModel.ndeps(::DiffResidual) = static(2)
NLLSModel.nres(::DiffResidual) = static(1)
NLLSModel.varindices(res::DiffResidual) = SVector(res.i, res.j)
NLLSModel.getvars(res::DiffResidual, vars::Vector) = (vars[res.i]::Float64, vars[res.j]::Float64)
NLLSModel.computeresidual(::DiffResidual, a, b) = a - b
Base.eltype(::DiffResidual) = Float64

# Measurements of three and four 2-vectors, for SimpleError3 and SimpleError4
const V2 = SVector{2, Float64}
NLLSModel.generatemeasurement(a::SVector{2}, b::SVector{2}, c::SVector{2}) = a + 2 * b - c
NLLSModel.generatemeasurement(a::SVector{2}, b::SVector{2}, c::SVector{2}, d::SVector{2}) = a + b + c + d

# A difference of two variables of either type
struct EitherDiff <: NLLSModel.AbstractResidual
    i::Int
    j::Int
end
NLLSModel.ndeps(::EitherDiff) = static(2)
NLLSModel.nres(::EitherDiff) = static(1)
NLLSModel.varindices(res::EitherDiff) = SVector(res.i, res.j)
NLLSModel.getvars(res::EitherDiff, vars::Vector) = (vars[res.i], vars[res.j])
NLLSModel.computeresidual(::EitherDiff, a, b) = a - b
Base.eltype(::EitherDiff) = Float64

# Residuals that break one rule of addcost! each (AN8)
struct BadRes{K} <: NLLSModel.AbstractResidual end
NLLSModel.ndeps(::BadRes) = static(1)
NLLSModel.ndeps(::BadRes{:ndeps}) = 1 # not static
NLLSModel.nres(::BadRes) = static(1)
NLLSModel.nres(::BadRes{:nres}) = static(0)
NLLSModel.varindices(::BadRes) = SVector(1)
NLLSModel.varindices(::BadRes{:varindices}) = SVector(1, 2)
NLLSModel.getvars(::BadRes, vars::Vector) = (vars[1]::Float64,)
NLLSModel.getvars(::BadRes{:getvars}, vars::Vector) = [vars[1]] # not a tuple
NLLSModel.computeresidual(::BadRes, a) = SVector(a)
Base.eltype(::BadRes) = Float64
struct BadAdaptive <: NLLSModel.AbstractAdaptiveResidual end
NLLSModel.ndeps(::BadAdaptive) = static(1)
NLLSModel.nres(::BadAdaptive) = static(1)
NLLSModel.varindices(::BadAdaptive) = SVector(1)
NLLSModel.getvars(::BadAdaptive, vars::Vector) = (vars[1]::Float64,) # not an adaptive robustifier
Base.eltype(::BadAdaptive) = Float64

module UnrollCheck
using NLLSModel
vars = () # A module-level global with the same name as the local in `count_unrolled`
function count_unrolled(vars)
    n = 0
    NLLSModel.@unroll for i in 1:3
        if i <= length(vars)
            n += 1
        end
    end
    return n
end
# Conditions on the loop variable alone are folded at expansion: the taken branch is kept, the other is dropped, and
# an `if` without `else` whose condition is false becomes `nothing`
function unrolled_branches()
    out = Int[]
    NLLSModel.@unroll for i in 1:3
        if i <= 2
            push!(out, i)
        else
            push!(out, -i)
        end
        if i == 3
            push!(out, 0)
        end
    end
    return out
end
end
