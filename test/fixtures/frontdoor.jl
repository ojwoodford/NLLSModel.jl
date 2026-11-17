# Pins a variable of any type to a target (nonlinear, with its minimum at the target)
struct Tgt{V} <: NLLSModel.AbstractResidual
    t::V
    i::Int
end
NLLSModel.ndeps(::Tgt) = static(1)
NLLSModel.nres(::Tgt{<:StaticVector{N}}) where N = static(N)
NLLSModel.nres(r::Tgt{<:Vector}) = length(r.t)
NLLSModel.varindices(r::Tgt) = r.i
NLLSModel.getvars(r::Tgt{V}, vars::AbstractVector) where V = (vars[r.i]::V,)
NLLSModel.computeresidual(r::Tgt, x) = (x .+ 0.1 .* x .^ 3) .- (r.t .+ 0.1 .* r.t .^ 3)
Base.eltype(::Tgt) = Float64

# Links two vectors of equal length: a - b = d
struct Lnk <: NLLSModel.AbstractResidual
    d::Vector{Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::Lnk) = static(2)
NLLSModel.nres(r::Lnk) = length(r.d)
NLLSModel.varindices(r::Lnk) = SVector(r.i, r.j)
NLLSModel.getvars(r::Lnk, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::Vector{Float64})
NLLSModel.computeresidual(r::Lnk, a, b) = a .- b .- r.d
Base.eltype(::Lnk) = Float64

const FDOptions = OptimizationOptions(iterator=:levenbergmarquardt, maxiters=100)

# One variable of type `V`, started away from its target
function singleproblem(V, n, seed)
    rng = MersenneTwister(seed)
    t = V(randn(rng, n))
    model = Model(V, Tgt{V})
    addvariable!(model, V(t .+ 0.3 .* randn(rng, n)))
    addcost!(model, Tgt(t, 1))
    return model, t
end

# A chain of `n` vectors of dimension `d`, each pinned to its target and linked to its neighbour
function chainproblem(n, d, seed)
    rng = MersenneTwister(seed)
    targets = [randn(rng, d) for _ in 1:n]
    model = Model(Vector{Float64}, Union{Tgt{Vector{Float64}}, Lnk})
    for k in 1:n
        addvariable!(model, targets[k] .+ 0.3 .* randn(rng, d))
    end
    for k in 1:n
        addcost!(model, Tgt(targets[k], k))
    end
    for k in 1:n-1
        addcost!(model, Lnk(targets[k] .- targets[k+1], k, k+1))
    end
    return model, targets
end

# Vectors of 3, 1, 2 dimensions and static 2-vectors, in an order that block_sizes_indices has to re-sort. Pairs (1, 4) and
# (3, 7) are linked, and no linked pair is split by any `unfixed` below, so every unfixed variable has its target as optimum.
function mixedproblem()
    rng = MersenneTwister(11)
    sizes = [3, 2, 1, 3, 2, 2, 1]
    isstatic = [false, true, false, false, true, false, false]
    targets = Any[isstatic[k] ? SVector{2, Float64}(randn(rng, 2)) : randn(rng, sizes[k]) for k in 1:7]
    model = Model(Union{Vector{Float64}, SVector{2, Float64}}, Union{Tgt{Vector{Float64}}, Tgt{SVector{2, Float64}}, Lnk})
    for k in 1:7
        addvariable!(model, targets[k] .+ 0.3 .* (isstatic[k] ? SVector{2, Float64}(randn(rng, 2)) : randn(rng, sizes[k])))
    end
    for k in 1:7
        addcost!(model, Tgt(targets[k], k))
    end
    addcost!(model, Lnk(targets[1] .- targets[4], 1, 4))
    addcost!(model, Lnk(targets[3] .- targets[7], 3, 7))
    return model, targets
end

# Pins a variable to a target linearly, with a hand-written Jacobian that allocates nothing (ForwardDiff on `Tgt` allocates
# for 16 or more dimensions and for a Vector), so an allocation test sees only the linear system's own. A dynamic one
# returns arrays it owns.
struct LinTgt{V, R, J} <: NLLSModel.AbstractResidual
    t::V
    r::R
    J::J
end
LinTgt(t::StaticVector{N}) where N = LinTgt(t, nothing, nothing)
LinTgt(t::Vector{Float64}) = LinTgt(t, similar(t), Matrix{Float64}(I, length(t), length(t)))
NLLSModel.ndeps(::LinTgt) = static(1)
NLLSModel.nres(::LinTgt{<:StaticVector{N}}) where N = static(N)
NLLSModel.nres(r::LinTgt{<:Vector}) = length(r.t)
NLLSModel.varindices(::LinTgt) = 1
NLLSModel.getvars(::LinTgt{V}, vars::AbstractVector) where V = (vars[1]::V,)
NLLSModel.computeresidual(r::LinTgt, x) = x - r.t
NLLSModel.computeresjac(::StaticInt{1}, r::LinTgt{<:StaticVector{N}}, x) where N = (x - r.t, SMatrix{N, N, Float64}(I))
NLLSModel.computeresjac(::StaticInt{1}, r::LinTgt{<:Vector}, x) = (r.r .= x .- r.t; (r.r, r.J))
Base.eltype(::LinTgt) = Float64

# Four costs, so that a shared system has a cost for each of up to four threads
function linsingleproblem(V, n, seed; ncosts=4)
    rng = MersenneTwister(seed)
    t = V(randn(rng, n))
    model = Model(V, typeof(LinTgt(t)))
    addvariable!(model, V(t .+ 0.3 .* randn(rng, n)))
    foreach(_ -> addcost!(model, LinTgt(V(t .+ 0.01 .* randn(rng, n)))), 1:ncosts)
    return model
end

# Allocation of one fdf! call, after a warm-up call (in a function, so the measurement sees concrete types)
singlefdf_allocs(fdf!, g, H, x) = (fdf!(g, H, x); @allocated fdf!(g, H, x))

landmarkvec(v) = v isa OptLandmark ? v.p : v
issolved(model, targets, which=eachindex(targets); atol=1e-6) = all(k -> isapprox(model.variables[k], targets[k]; atol=atol), which)
