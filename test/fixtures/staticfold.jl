# Models whose single unfixed variable has a static system, one per kind of per-cost Hessian it can be handed, for the
# tests of the fold (staticfold.jl, staticfold_threaded.jl); and the in-place accumulation of the same costs into a
# dynamic system, the reference.
using NLLSModel, StaticArrays, Static, LinearAlgebra
import OptimizationVariableInterface as OVI

# Deterministic data in [-1, 1)
sf_data(k, i) = sin(1.7 * k + 0.37 * i + 0.1 * k * i)
sf_vec(::Val{N}, k) where N = SVector{N, Float64}(ntuple(i -> sf_data(k, i), Val(N)))

# A scalar residual ½‖x − p‖² − d with a hand-written Jacobian, the adjoint of an SVector; and the same with a kernel
struct SFScalar{N, K} <: AbstractResidual
    p::SVector{N, Float64}
    d::Float64
    kernel::K
end
NLLSModel.ndeps(::SFScalar) = static(1)
NLLSModel.nres(::SFScalar) = static(1)
NLLSModel.varindices(::SFScalar) = 1
NLLSModel.getvars(::SFScalar{N}, vars::AbstractVector) where N = (vars[1]::SVector{N, Float64},)
NLLSModel.robustkernel(r::SFScalar) = r.kernel
Base.eltype(::SFScalar) = Float64
NLLSModel.computeresidual(r::SFScalar, x) = 0.5 * sum(abs2, x - r.p) - r.d
function NLLSModel.computeresjac(::StaticInt{1}, r::SFScalar, x)
    u = x - r.p
    return 0.5 * dot(u, u) - r.d, u'
end

# M residual rows with a hand-written static Jacobian (an SMatrix{M, N}), optionally a kernel
struct SFRows{N, M, K, L} <: AbstractResidual
    P::SMatrix{M, N, Float64, L}
    kernel::K
end
NLLSModel.ndeps(::SFRows) = static(1)
NLLSModel.nres(::SFRows{N, M}) where {N, M} = static(M)
NLLSModel.varindices(::SFRows) = 1
NLLSModel.getvars(::SFRows{N}, vars::AbstractVector) where N = (vars[1]::SVector{N, Float64},)
NLLSModel.robustkernel(r::SFRows) = r.kernel
Base.eltype(::SFRows) = Float64
NLLSModel.computeresidual(r::SFRows{N, M}, x) where {N, M} = r.P * x + SVector{M}(ntuple(k -> 0.1 * sum(abs2, x) - 0.3k, Val(M)))
function NLLSModel.computeresjac(::StaticInt{1}, r::SFRows{N, M}, x) where {N, M}
    J = r.P + SMatrix{M, N}(ntuple(l -> 0.2 * x[(l - 1) ÷ M + 1], Val(M * N)))
    return NLLSModel.computeresidual(r, x), J
end

# Differentiated by NLLSModel (no computeresjac): static duals for M·N ≤ MAX_STATIC_JAC, the wide path (a Matrix
# Jacobian, the static-column kernel) above
struct SFAuto{N, M, L} <: AbstractResidual
    P::SMatrix{M, N, Float64, L}
end
NLLSModel.ndeps(::SFAuto) = static(1)
NLLSModel.nres(::SFAuto{N, M}) where {N, M} = static(M)
NLLSModel.varindices(::SFAuto) = 1
NLLSModel.getvars(::SFAuto{N}, vars::AbstractVector) where N = (vars[1]::SVector{N, Float64},)
Base.eltype(::SFAuto) = Float64
NLLSModel.computeresidual(r::SFAuto{N, M}, x) where {N, M} = r.P * x + SVector{M}(ntuple(k -> 0.05 * k * sum(abs2, x), Val(M)))

# A cost that is not a squared residual (its Hessian by ForwardDiff: a full static matrix)
struct SFCost{N} <: AbstractCost
    p::SVector{N, Float64}
end
NLLSModel.ndeps(::SFCost) = static(1)
NLLSModel.varindices(::SFCost) = 1
NLLSModel.getvars(::SFCost{N}, vars::AbstractVector) where N = (vars[1]::SVector{N, Float64},)
Base.eltype(::SFCost) = Float64
NLLSModel.computecost(c::SFCost, x) = log1p(0.5 * sum(abs2, x - c.p))

# A residual on the unfixed variable and K fixed ones (value dispatch on the varflags for K + 1 ≤ 5, run-time dispatch
# above); the variables are scalars or SVector{3}s
struct SFDeps{K, V} <: AbstractResidual
    vi::SVector{K, Int}
    d::Float64
end
NLLSModel.ndeps(::SFDeps{K}) where K = static(K)
NLLSModel.nres(::SFDeps) = static(1)
NLLSModel.varindices(r::SFDeps) = r.vi
NLLSModel.getvars(r::SFDeps{K, V}, vars::AbstractVector) where {K, V} = ntuple(i -> vars[r.vi[i]]::V, Val(K))
Base.eltype(::SFDeps) = Float64
NLLSModel.computeresidual(r::SFDeps, xs...) = sum(x -> 0.5 * sum(abs2, x), xs) - r.d

# An adaptive kernel as the only unfixed variable (the robust-kernel fit's shape): the kernel-only branch, a full
# static matrix
struct SFAdaptive <: AbstractAdaptiveResidual
    p::SVector{2, Float64}
    vi::Int
end
NLLSModel.ndeps(::SFAdaptive) = static(2)
NLLSModel.nres(::SFAdaptive) = static(1)
NLLSModel.varindices(r::SFAdaptive) = SVector(1, r.vi)
const SFKernel = typeof(ContaminatedGaussian(0.3, 3.0, 0.7, static(1)))
NLLSModel.getvars(r::SFAdaptive, vars::AbstractVector) = (vars[1]::SFKernel, vars[r.vi]::SVector{2, Float64})
Base.eltype(::SFAdaptive) = Float64
NLLSModel.computeresidual(r::SFAdaptive, x) = 0.5 * sum(abs2, x - r.p) - 0.4

function sf_model(vars::Vector, costs::Vector...)
    model = Model(eltype(vars), Union{map(eltype, costs)...})
    foreach(v -> addvariable!(model, v), vars)
    for v in costs, c in v
        addcost!(model, c)
    end
    return model
end

scalarmodel(N, nc; kernel=NoRobust()) =
    (sf_model([sf_vec(Val(N), 0)], [SFScalar{N, typeof(kernel)}(sf_vec(Val(N), k), 0.3 * sf_data(k, 9), kernel) for k in 1:nc]), nothing)
function rowsmodel(N, M, nc; kernel=NoRobust())
    costs = [SFRows{N, M, typeof(kernel), M * N}(SMatrix{M, N}(ntuple(l -> sf_data(k, l), Val(M * N))), kernel) for k in 1:nc]
    return sf_model([sf_vec(Val(N), 0)], costs), nothing
end
automodel(N, M, nc) = (sf_model([sf_vec(Val(N), 0)], [SFAuto{N, M, M * N}(SMatrix{M, N}(ntuple(l -> sf_data(k, l), Val(M * N)))) for k in 1:nc]), nothing)
costmodel(N, nc) = (sf_model([sf_vec(Val(N), 0)], [SFCost{N}(sf_vec(Val(N), k)) for k in 1:nc]), nothing)
# Two cost types, one with no costs: an empty cost vector, then two non-empty ones
function mixedmodel(nc)
    model = Model(SVector{3, Float64}, Union{SFCost{3}, SFScalar{3, NoRobust}, SFRows{3, 2, NoRobust, 6}})
    addvariable!(model, sf_vec(Val(3), 0))
    for k in 1:nc
        addcost!(model, SFScalar{3, NoRobust}(sf_vec(Val(3), k), 0.3, NoRobust()))
        isodd(k) && addcost!(model, SFRows{3, 2, NoRobust, 6}(SMatrix{2, 3}(ntuple(l -> sf_data(k, l), Val(6))), NoRobust()))
    end
    return model, nothing
end
# A scalar variable (a 1 × 1 static system)
function onemodel(nc)
    model = Model(Float64, SFDeps{1, Float64})
    addvariable!(model, 0.7)
    foreach(k -> addcost!(model, SFDeps{1, Float64}(SVector(1), sf_data(k, 1))), 1:nc)
    return model, nothing
end
# The unfixed SVector{3} (variable 1) with K − 1 fixed ones
function depsmodel(K, nc)
    vars = [sf_vec(Val(3), i) for i in 1:K]
    model = Model(SVector{3, Float64}, SFDeps{K, SVector{3, Float64}})
    foreach(v -> addvariable!(model, v), vars)
    foreach(k -> addcost!(model, SFDeps{K, SVector{3, Float64}}(SVector{K}(ntuple(i -> mod1(i + k - 1, K), Val(K))), sf_data(k, 2))), 1:nc)
    return model, 1
end
function adaptivemodel(nc)
    model = Model(Union{SFKernel, SVector{2, Float64}}, SFAdaptive)
    addvariable!(model, ContaminatedGaussian(0.3, 3.0, 0.7, static(1)))
    foreach(i -> addvariable!(model, sf_vec(Val(2), i)), 1:4)
    foreach(k -> addcost!(model, SFAdaptive(sf_vec(Val(2), k + 10), 2 + mod(k, 4))), 1:nc)
    return model, 1
end
# A cost on variable 2 only, with variable 1 unfixed: a cost with no unfixed variable
function nounfixedmodel()
    model = Model(SVector{3, Float64}, SFDeps{1, SVector{3, Float64}})
    addvariable!(model, sf_vec(Val(3), 1)); addvariable!(model, sf_vec(Val(3), 2))
    addcost!(model, SFDeps{1, SVector{3, Float64}}(SVector(1), 0.1))
    addcost!(model, SFDeps{1, SVector{3, Float64}}(SVector(2), 0.2))
    return model, 1
end

# (name, builder, precision, whether an evaluation allocates nothing); every builder returns (model, unfixed). More than
# five dependencies with a fixed one are dispatched at run time, which allocates on the in-place path too.
const STATICFOLD_CASES = [
    ("scalar residual, N = 3", () -> scalarmodel(3, 40), Float64, true),
    ("scalar residual, N = 8, Float32", () -> scalarmodel(8, 40), Float32, true),
    ("scalar residual, N = 12 (MAX_STATIC_SYSTEM)", () -> scalarmodel(12, 20), Float64, true),
    ("scalar residual, Huber2oKernel (both arms)", () -> scalarmodel(3, 40; kernel=Huber2oKernel(0.2)), Float64, true),
    ("3 rows, N = 3", () -> rowsmodel(3, 3, 30), Float64, true),
    ("3 rows, N = 3, Float32", () -> rowsmodel(3, 3, 30), Float32, true),
    ("2 rows, N = 5, Huber2oKernel", () -> rowsmodel(5, 2, 30; kernel=Huber2oKernel(0.5)), Float64, true),
    ("autodiff, N = 4, M = 2", () -> automodel(4, 2, 30), Float64, true),
    ("autodiff, wide (Matrix Jacobian), N = 12, M = 11", () -> automodel(12, 11, 10), Float64, true),
    ("a cost that is not a residual", () -> costmodel(3, 30), Float64, true),
    ("several cost types, one empty", () -> mixedmodel(20), Float64, true),
    ("a scalar variable", () -> onemodel(30), Float64, VERSION >= v"1.11"), # on 1.10 the in-place path allocates too
    ("fixed dependencies (value dispatch)", () -> depsmodel(3, 30), Float64, true),
    ("fixed dependencies (run-time dispatch, 6 dependencies)", () -> depsmodel(6, 30), Float64, false),
    ("adaptive kernel alone unfixed", () -> adaptivemodel(30), Float64, true),
]

# The in-place accumulation of the same costs into a dynamic single-variable system: cost, g and H
function inplace_reference(model, unfixed, N, ::Type{T}) where T
    u = unfixed === nothing ? 1 : unfixed
    ls = NLLSModel.LinearSystem{NLLSModel.UniVariateLSdynamic_x, NLLSModel.UniVariateLSdynamic_ls{T}}((u, N), (N,))
    c = NLLSModel.costgradhess!(ls, model.variables, model.costs)
    return c, copy(NLLSModel.getb(ls)), copy(NLLSModel.getA(ls))
end

# The types of the per-cost Hessians of a model's costs
function kindsof(model, unfixed)
    u = unfixed === nothing ? 1 : unfixed
    ls = NLLSModel.LinearSystem{NLLSModel.UniVariateLSdynamic_x, NLLSModel.UniVariateLSdynamic_ls{Float64}}((u, 1), (1,))
    kinds = Set{Type}()
    for v in values(model.costs), cost in v
        vf = NLLSModel.computevarflags(NLLSModel.getoffsets(cost, ls))
        vf == 0 && continue
        push!(kinds, typeof(NLLSModel.computecostgradhess(static(vf), cost, NLLSModel.getvars(cost, model.variables)...)[3]))
    end
    return kinds
end
