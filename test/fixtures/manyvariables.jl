# Each variable near a target
struct ManyPrior{V} <: NLLSModel.AbstractResidual
    t::SVector{2, Float64}
    i::Int
end
NLLSModel.ndeps(::ManyPrior) = static(1)
NLLSModel.nres(::ManyPrior) = static(2)
NLLSModel.varindices(r::ManyPrior) = r.i
NLLSModel.getvars(r::ManyPrior{V}, vars::AbstractVector) where V = (vars[r.i]::V,)
NLLSModel.computeresidual(r::ManyPrior, x) = SVector(x[1] - r.t[1], x[2] - r.t[2])
Base.eltype(::ManyPrior) = Float64

# One cost of N variables, 3 rows, nonlinear in each
struct ManyRes{N, V} <: NLLSModel.AbstractResidual
    z::SVector{3, Float64}
    inds::SVector{N, Int}
end
NLLSModel.ndeps(::ManyRes{N}) where N = static(N)
NLLSModel.nres(::ManyRes) = static(3)
NLLSModel.varindices(r::ManyRes) = r.inds
NLLSModel.getvars(r::ManyRes{N, V}, vars::AbstractVector) where {N, V} = ntuple(k -> vars[r.inds[k]]::V, Val(N))
manyres(xs) = SVector(ntuple(j -> sum(k -> sin(0.5 * j * xs[k][1]) + xs[k][2] * (1 + 0.1 * xs[k][2]) / (k + j), 1:length(xs)), Val(3)))
NLLSModel.computeresidual(r::ManyRes, xs...) = manyres(xs) - r.z
Base.eltype(::ManyRes) = Float64

makevar(::Type{SVector{2, Float64}}, v) = SVector{2, Float64}(v)
makevar(::Type{Vector{Float64}}, v) = Vector{Float64}(v)

# N variables, a prior on each, and two costs of all N variables (in different orders)
function manymodel(N, ::Type{V}; seed=7) where V
    rng = MersenneTwister(seed)
    model = Model(V, Union{ManyPrior{V}, ManyRes{N, V}})
    truth = [randn(rng, SVector{2, Float64}) for _ in 1:N]
    for k in 1:N
        addvariable!(model, makevar(V, truth[k] + 0.2 * randn(rng, SVector{2, Float64})))
        addcost!(model, ManyPrior{V}(truth[k] + 0.05 * randn(rng, SVector{2, Float64}), k))
    end
    for inds in (SVector{N}(1:N), SVector{N}(reverse(1:N)))
        addcost!(model, ManyRes{N, V}(manyres(Tuple(truth[inds])) + 0.05 * randn(rng, SVector{3, Float64}), inds))
    end
    return model
end
