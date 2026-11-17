using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
using SecondOrderOptimization

# AN1: a cost with more than 5 variables, some of them fixed (no value dispatch on which: the variable flags are a run-time
# value), on both paths. Variables of static size (SVector{2}) and of run-time size (Vector{Float64} of length 2).

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

@testset "manyvariables.jl" begin
    nt = Threads.nthreads()
    opts = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=200)
    @testset "a cost of $N variables, $V, unfixed = $unfixed" for N in (6, 10), V in (SVector{2, Float64}, Vector{Float64}),
            unfixed in (collect(2:N), collect(3:2:N), [N])
        model = manymodel(N, V)
        for numthreads in unique((1, nt))
            # The tall linearization equals the normal one, for the same unfixed variables
            _, _, resjac!, xt, J = optimvars_nls(model; unfixed, numthreads=static(numthreads), reduction=:none, storage=:dense, min_work=0, min_work_cost=0)
            r = zeros(size(J, 1))
            _, fdf!, xn, _ = optimvars(model; unfixed, numthreads=static(numthreads), reduction=:none, storage=:dense, min_work=0, min_work_cost=0)
            g, H = zeros(size(J, 2)), zeros(size(J, 2), size(J, 2))
            c = fdf!(g, H, xn)
            @test resjac!(r, J, xt) ≈ c rtol=1e-12
            @test J' * r ≈ g rtol=1e-9 atol=1e-12
            @test J' * J ≈ Symmetric(H) rtol=1e-9 atol=1e-12
            # The tall solve reaches the normal one's cost
            normal = deepcopy(model)
            optimize!(normal, opts; unfixed, numthreads=static(numthreads))
            tall = deepcopy(model)
            optimize!(tall, opts; formulation=:tall, unfixed, numthreads=static(numthreads))
            @test NLLSModel.cost(normal) < NLLSModel.cost(model)
            @test NLLSModel.cost(tall) ≈ NLLSModel.cost(normal) rtol=1e-8
        end
    end
end
