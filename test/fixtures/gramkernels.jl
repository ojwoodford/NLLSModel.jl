# ---- computerescostgradhess against the previous implementation (BLAS products and reweighting passes) ----
function reference_costgradhess(varflags, residual, kernel, vars)
    varflagsres = varflags >> static(1)
    res, jac = NLLSModel.computeresjac(varflagsres, residual, vars...)
    sqres = NLLSModel.sqnorm(res)
    g = jac' * res
    H = jac' * jac
    if varflags & 1 == 0
        cost, dc, d2c = robustifydcost(kernel, sqres)
    else
        cost, dc_, d2c_ = robustifydkernel(kernel, sqres)
        dc = dc_[end]
        d2c = d2c_[end, end]
        kernelvarind = NLLSModel.SR(1, NLLSModel.nvars(kernel))
        dkdv = g * view(d2c_, kernelvarind, NLLSModel.nvars(kernel) + 1)'
    end
    dc != 1 && (H *= dc)
    d2c != 0 && (H += (max(2 * d2c, -dc / (3 * sqres)) * g) * g')
    dc != 1 && (g *= dc)
    if varflags & 1 == 1
        g = vcat(0.5 * view(dc_, kernelvarind), g)
        H = hcat(vcat(0.5 * view(d2c_, kernelvarind, kernelvarind), dkdv), vcat(dkdv', H))
    end
    return 0.5 * Float64(cost), g, H
end

# Residual of a variable with run-time size: M entries x[k] * x[k+1] - meas[k], as an SVector (static length) or a Vector
struct GramRes{M, K} <: NLLSModel.AbstractResidual # M == 0: run-time length
    meas::Vector{Float64}
    kernel::K
end
NLLSModel.ndeps(::GramRes) = static(1)
NLLSModel.nres(r::GramRes{M}) where M = M == 0 ? length(r.meas) : static(M)
NLLSModel.varindices(::GramRes) = 1
NLLSModel.getvars(::GramRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.robustkernel(r::GramRes) = r.kernel
Base.eltype(::GramRes) = Float64
NLLSModel.computeresidual(r::GramRes{0}, x) = [x[k] * x[k+1] - r.meas[k] for k in eachindex(r.meas)]
NLLSModel.computeresidual(r::GramRes{M}, x) where M = SVector{M}(ntuple(k -> x[k] * x[k+1] - r.meas[k], Val(M)))

# The same with a scalar residual
struct GramScalar{K} <: NLLSModel.AbstractResidual
    meas::Float64
    kernel::K
end
NLLSModel.ndeps(::GramScalar) = static(1)
NLLSModel.nres(::GramScalar) = static(1)
NLLSModel.varindices(::GramScalar) = 1
NLLSModel.getvars(::GramScalar, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.robustkernel(r::GramScalar) = r.kernel
Base.eltype(::GramScalar) = Float64
NLLSModel.computeresidual(r::GramScalar, x) = sum(abs2, x) - r.meas

# Adaptive-kernel residual of a run-time sized variable
struct GramAdaptive{M} <: NLLSModel.AbstractAdaptiveResidual
    meas::Vector{Float64}
end
NLLSModel.ndeps(::GramAdaptive) = static(2)
NLLSModel.nres(::GramAdaptive{M}) where M = static(M)
NLLSModel.varindices(::GramAdaptive) = SVector(1, 2)
NLLSModel.getvars(::GramAdaptive, vars::AbstractVector) = (vars[1]::ContaminatedGaussian{Float64}, vars[2]::Vector{Float64})
Base.eltype(::GramAdaptive) = Float64
NLLSModel.computeresidual(r::GramAdaptive{M}, x) where M = SVector{M}(ntuple(k -> x[k] * x[k+1] - r.meas[k], Val(M)))

struct GramAdaptiveHand{M} <: NLLSModel.AbstractAdaptiveResidual
    meas::Vector{Float64}
    res::Vector{Float64}
    jac::Matrix{Float64}
end
NLLSModel.ndeps(::GramAdaptiveHand) = static(2)
NLLSModel.nres(::GramAdaptiveHand{M}) where M = static(M)
NLLSModel.varindices(::GramAdaptiveHand) = SVector(1, 2)
NLLSModel.getvars(::GramAdaptiveHand, vars::AbstractVector) = (vars[1]::ContaminatedGaussian{Float64}, vars[2]::Vector{Float64})
Base.eltype(::GramAdaptiveHand) = Float64
function NLLSModel.computeresjac(varflags, r::GramAdaptiveHand{M}, x) where M
    for k in 1:M
        r.res[k] = x[k] * x[k+1] - r.meas[k]
        fill!(view(r.jac, k, :), 0.0)
        r.jac[k, k] = x[k+1]
        r.jac[k, k+1] = x[k]
    end
    return r.res, r.jac
end

# Residuals with hand-written Jacobians in buffers they own (as a user with dynamic variables would write them). The
# autodiff path of a `Vector` variable allocates in `update`, which builds a dual vector, so it is not used here.
struct GramHand{M, K} <: NLLSModel.AbstractResidual # M == 0: run-time length
    meas::Vector{Float64}
    res::Vector{Float64}
    jac::Matrix{Float64}
    kernel::K
end
GramHand{M}(meas, n, kernel) where M = GramHand{M, typeof(kernel)}(meas, zeros(length(meas)), zeros(max(length(meas), 1), n), kernel)
NLLSModel.ndeps(::GramHand) = static(1)
NLLSModel.nres(r::GramHand{M}) where M = M == 0 ? length(r.meas) : static(M)
NLLSModel.varindices(::GramHand) = 1
NLLSModel.getvars(::GramHand, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.robustkernel(r::GramHand) = r.kernel
Base.eltype(::GramHand) = Float64
function NLLSModel.computeresjac(varflags, r::GramHand{M}, x) where M
    m = length(r.meas)
    for k in 1:m
        r.res[k] = x[k] * x[k+1] - r.meas[k]
        fill!(view(r.jac, k, :), 0.0)
        r.jac[k, k] = x[k+1]
        r.jac[k, k+1] = x[k]
    end
    return r.res, r.jac
end

# Product cap of the full static path (MAX_STATIC_JAC)
struct CapRes{M, N} <: NLLSModel.AbstractResidual # N static variables of size N, M residuals
    meas::SVector{M, Float64}
end
NLLSModel.ndeps(::CapRes) = static(1)
NLLSModel.nres(::CapRes{M}) where M = static(M)
NLLSModel.varindices(::CapRes) = 1
NLLSModel.getvars(::CapRes{M, N}, vars::AbstractVector) where {M, N} = (vars[1]::SVector{N, Float64},)
Base.eltype(::CapRes) = Float64
NLLSModel.computeresidual(r::CapRes{M, N}, x) where {M, N} = SVector{M}(ntuple(k -> x[mod1(k, N)] * x[mod1(k + 1, N)] - r.meas[k], Val(M)))
# A residual whose length is only known at run time
struct CapResDynamic <: NLLSModel.AbstractResidual
    meas::Vector{Float64}
end
NLLSModel.ndeps(::CapResDynamic) = static(1)
NLLSModel.nres(r::CapResDynamic) = length(r.meas)
NLLSModel.varindices(::CapResDynamic) = 1
NLLSModel.getvars(::CapResDynamic, vars::AbstractVector) = (vars[1]::SVector{4, Float64},)
Base.eltype(::CapResDynamic) = Float64
NLLSModel.computeresidual(r::CapResDynamic, x) = SVector{4}(ntuple(k -> x[mod1(k, 4)] * x[mod1(k + 1, 4)] - r.meas[k], Val(4)))

function fdjacobian(res, x; h=1e-6)
    cols = map(1:length(x)) do i
        e = SVector{length(x), Float64}(ntuple(j -> j == i ? h : 0.0, length(x)))
        collect(NLLSModel.computeresidual(res, x + e) - NLLSModel.computeresidual(res, x - e)) / 2h
    end
    return reduce(hcat, cols)
end

# ---- B1: a static residual with a dynamic Jacobian ----
# Hand-written computeresjac returning a residual of container type `R` (SVector, MVector, Vector) and a `Matrix` Jacobian
struct StatHand{M, R, K} <: NLLSModel.AbstractResidual # R: :svector, :mvector or :vector; nres is static unless `dynamicnres`
    meas::Vector{Float64}
    jac::Matrix{Float64}
    kernel::K
    dynamicnres::Bool
end
StatHand{M, R}(meas, n, kernel; dynamicnres=false) where {M, R} = StatHand{M, R, typeof(kernel)}(meas, zeros(M, n), kernel, dynamicnres)
NLLSModel.ndeps(::StatHand) = static(1)
NLLSModel.nres(r::StatHand{M}) where M = r.dynamicnres ? M : static(M)
NLLSModel.varindices(::StatHand) = 1
NLLSModel.getvars(::StatHand, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.robustkernel(r::StatHand) = r.kernel
Base.eltype(::StatHand) = Float64
function NLLSModel.computeresjac(varflags, r::StatHand{M, R}, x) where {M, R}
    for k in 1:M
        fill!(view(r.jac, k, :), 0.0)
        r.jac[k, k] = x[k+1]
        r.jac[k, k+1] = x[k]
    end
    vals = ntuple(k -> x[k] * x[k+1] - r.meas[k], Val(M))
    res = R === :svector ? SVector{M, Float64}(vals) : R === :mvector ? MVector{M, Float64}(vals) : collect(vals)
    return res, r.jac
end

# The same with the kernel parameters of an adaptive kernel as variable 1
struct StatAdaptive{M, R} <: NLLSModel.AbstractAdaptiveResidual
    meas::Vector{Float64}
    jac::Matrix{Float64}
end
StatAdaptive{M, R}(meas, n::Integer) where {M, R} = StatAdaptive{M, R}(meas, zeros(M, n))
NLLSModel.ndeps(::StatAdaptive) = static(2)
NLLSModel.nres(::StatAdaptive{M}) where M = static(M)
NLLSModel.varindices(::StatAdaptive) = SVector(1, 2)
NLLSModel.getvars(::StatAdaptive, vars::AbstractVector) = (vars[1]::ContaminatedGaussian{Float64}, vars[2]::Vector{Float64})
Base.eltype(::StatAdaptive) = Float64
function NLLSModel.computeresjac(varflags, r::StatAdaptive{M, R}, x) where {M, R}
    for k in 1:M
        fill!(view(r.jac, k, :), 0.0)
        r.jac[k, k] = x[k+1]
        r.jac[k, k+1] = x[k]
    end
    vals = ntuple(k -> x[k] * x[k+1] - r.meas[k], Val(M))
    return (R === :svector ? SVector{M, Float64}(vals) : MVector{M, Float64}(vals)), r.jac
end

# One static variable of size N, with the Jacobian returned as an SMatrix (S) or as a Matrix
struct StatFull{M, N, S, K} <: NLLSModel.AbstractResidual
    meas::SVector{M, Float64}
    kernel::K
end
NLLSModel.ndeps(::StatFull) = static(1)
NLLSModel.nres(::StatFull{M}) where M = static(M)
NLLSModel.varindices(::StatFull) = 1
NLLSModel.getvars(::StatFull{M, N}, vars::AbstractVector) where {M, N} = (vars[1]::SVector{N, Float64},)
NLLSModel.robustkernel(r::StatFull) = r.kernel
Base.eltype(::StatFull) = Float64
function NLLSModel.computeresjac(varflags, r::StatFull{M, N, S}, x) where {M, N, S}
    res = SVector{M, Float64}(ntuple(k -> x[mod1(k, N)] * x[mod1(k + 1, N)] - r.meas[k], Val(M)))
    J = zeros(M, N)
    for k in 1:M
        J[k, mod1(k, N)] += x[mod1(k + 1, N)]
        J[k, mod1(k + 1, N)] += x[mod1(k, N)]
    end
    return res, S ? SMatrix{M, N, Float64}(J) : J
end

# A residual whose `nres` is whatever it is told
struct NresOf{V}
    v::V
end
NLLSModel.nres(r::NresOf) = r.v

# The kernel `gramgrad!` reaches, from the predicates it dispatches on
function kernelof(res, jac)
    n = size(jac, 2)
    return NLLSModel.gram_kernel(zeros(n, n), zeros(n), jac, res)
end

# The Procrustes "Specialized" layout: one variable, a static 7-vector residual per point, the Jacobian written into a per-task Matrix
struct ProcRes{HAND} <: NLLSModel.AbstractResidual
    p::SVector{4, Float64}
    q::SVector{7, Float64}
end
NLLSModel.ndeps(::ProcRes) = static(1)
NLLSModel.nres(::ProcRes) = static(7)
NLLSModel.varindices(::ProcRes) = 1
NLLSModel.getvars(::ProcRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
Base.eltype(::ProcRes) = Float64
procvalue(r::ProcRes, a, k) = sin(a[mod1(k, 4)]) * r.p[mod1(k + 1, 4)] + 0.1 * a[mod1(k + 1, 4)]^2 - r.q[k]
NLLSModel.computeresidual(r::ProcRes, a) = SVector{7}(ntuple(k -> procvalue(r, a, k), Val(7)))
function NLLSModel.computeresjac(varflags, r::ProcRes{true}, a)
    J = get!(() -> zeros(7, 4), task_local_storage(), :procrustes_jac)::Matrix{Float64}
    fill!(J, 0.0)
    for k in 1:7
        J[k, mod1(k, 4)] += cos(a[mod1(k, 4)]) * r.p[mod1(k + 1, 4)]
        J[k, mod1(k + 1, 4)] += 0.2 * a[mod1(k + 1, 4)]
    end
    return SVector{7}(ntuple(k -> procvalue(r, a, k), Val(7))), J
end

function procproblem(HAND, npoints, seed)
    rng = MersenneTwister(seed)
    truth = [0.3, -0.5, 0.8, 0.2]
    model = Model(Vector{Float64}, ProcRes{HAND})
    addvariable!(model, truth .+ 0.4 .* randn(rng, 4))
    for _ in 1:npoints
        p = SVector{4, Float64}(randn(rng, 4))
        q = SVector{7, Float64}(ntuple(k -> procvalue(ProcRes{HAND}(p, zeros(SVector{7})), truth, k), Val(7))) .+ 0.01 .* randn(rng, SVector{7, Float64})
        addcost!(model, ProcRes{HAND}(p, q))
    end
    return model
end
