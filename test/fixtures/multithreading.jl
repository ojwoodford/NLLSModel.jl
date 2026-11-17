myfun(x) = x * x * x
NLLSModel.computecost(::StaticInt, ::AbstractVector, x::Float64) = myfun(x)
NLLSModel.computecost(::AbstractVector, x::Float64) = myfun(x)

struct MTTestScalarRes <: NLLSModel.AbstractResidual
    val::Float64
    ind::Int
end
NLLSModel.ndeps(::MTTestScalarRes) = static(1)
NLLSModel.nres(::MTTestScalarRes) = 1
NLLSModel.varindices(res::MTTestScalarRes) = res.ind
NLLSModel.getvars(res::MTTestScalarRes, vars::AbstractVector) = (vars[res.ind]::Float64,)
NLLSModel.computeresidual(res::MTTestScalarRes, x) = x - res.val
Base.eltype(::MTTestScalarRes) = Float64

struct MTTestPairRes <: NLLSModel.AbstractResidual
    val::Float64
    i::Int
    j::Int
end
NLLSModel.ndeps(::MTTestPairRes) = static(2)
NLLSModel.nres(::MTTestPairRes) = static(1)
NLLSModel.varindices(res::MTTestPairRes) = SVector(res.i, res.j)
NLLSModel.getvars(res::MTTestPairRes, vars::AbstractVector) = (vars[res.i]::Float64, vars[res.j]::Float64)
NLLSModel.computeresidual(res::MTTestPairRes, x, y) = SVector(x - y - res.val)
Base.eltype(::MTTestPairRes) = Float64

struct MTTestDynamicRes <: NLLSModel.AbstractResidual
    y::Float64
    X::Vector{Float64}
end
NLLSModel.ndeps(::MTTestDynamicRes) = static(1)
NLLSModel.nres(::MTTestDynamicRes) = static(1)
NLLSModel.varindices(::MTTestDynamicRes) = 1
NLLSModel.getvars(::MTTestDynamicRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(res::MTTestDynamicRes, w) = dot(res.X, w) - res.y
Base.eltype(::MTTestDynamicRes) = Float64

alloc_ephemeral(ls, ind) = @allocated NLLSModel.LinearSystem(ls, ind)

# Bundle-adjustment shape: 2 rows on a 9-dimensional camera and a 3-dimensional point (12 columns)
struct WBARes <: NLLSModel.AbstractResidual
    meas::SVector{2, Float64}
    cam::Int
    pt::Int
end
NLLSModel.ndeps(::WBARes) = static(2)
NLLSModel.nres(::WBARes) = static(2)
NLLSModel.varindices(r::WBARes) = SVector(r.cam, r.pt)
NLLSModel.getvars(r::WBARes, vars::AbstractVector) = (vars[r.cam]::SVector{9, Float64}, vars[r.pt]::SVector{3, Float64})
const WBA_EVALS = Threads.Atomic{Int}(0) # residual evaluations, for counting how often a cost is linearized
function NLLSModel.computeresidual(r::WBARes, c, p)
    Threads.atomic_add!(WBA_EVALS, 1)
    return SVector(c[1] * p[1] + sin(c[5]) * p[2] + c[9] * p[3]^2, c[2] * p[2] - c[3] * p[1] + c[7]) - r.meas
end
Base.eltype(::WBARes) = Float64

function wbaproblem(ncams, npts)
    p = Model(Union{SVector{9, Float64}, SVector{3, Float64}}, WBARes)
    for i in 1:ncams
        addvariable!(p, SVector{9, Float64}(ntuple(k -> 0.1 * i + 0.01 * k, 9)))
    end
    for j in 1:npts
        addvariable!(p, SVector(0.3 * j, -0.2 * j, 1.0 + 0.05 * j))
    end
    for i in 1:ncams, j in 1:npts
        addcost!(p, WBARes(SVector(0.1 * i - 0.02 * j, 0.05 * j), i, ncams + j))
    end
    return p
end

# Procrustes shape: d rows on one rotation of d(d-1)/2 parameters
struct WProcRes{D} <: NLLSModel.AbstractResidual
    a::Vector{Float64}
    b::Vector{Float64}
end
NLLSModel.ndeps(::WProcRes) = static(1)
NLLSModel.nres(::WProcRes{D}) where D = static(D)
NLLSModel.varindices(::WProcRes) = 1
NLLSModel.getvars(::WProcRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
function wrotate(x, b, d)
    W = zeros(eltype(x), d, d); k = 1
    for j in 1:d, i in j+1:d
        W[i, j] = x[k]; W[j, i] = -x[k]; k += 1
    end
    return b .+ W * b .+ 0.5 .* (W * (W * b))
end
NLLSModel.computeresidual(r::WProcRes{D}, x) where D = r.a .- wrotate(x, r.b, D)
Base.eltype(::WProcRes) = Float64

function wprocrustes(d, npts)
    p = Model(Vector{Float64}, WProcRes{d})
    addvariable!(p, [0.01 * sin(k) for k in 1:d * (d - 1) ÷ 2])
    for k in 1:npts
        b = [cos(k + i) for i in 1:d]
        addcost!(p, WProcRes{d}(b .+ 0.01 .* sin.(1:d), b))
    end
    return p
end

# Dynamic variables of different lengths; a run-time residual length
struct WDynRes <: NLLSModel.AbstractResidual
    meas::Vector{Float64}
    i::Int
    j::Int
end
NLLSModel.ndeps(::WDynRes) = static(2)
NLLSModel.nres(r::WDynRes) = length(r.meas)
NLLSModel.varindices(r::WDynRes) = SVector(r.i, r.j)
NLLSModel.getvars(r::WDynRes, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::Vector{Float64})
NLLSModel.computeresidual(r::WDynRes, a, b) = [sum(a) * (1 + 0.1k) - b[mod1(k, length(b))]^2 - r.meas[k] for k in eachindex(r.meas)]
Base.eltype(::WDynRes) = Float64

function wdynproblem()
    p = Model(Vector{Float64}, WDynRes)
    addvariable!(p, [0.1, 0.2])            # 2
    addvariable!(p, [0.3, -0.1, 0.4])      # 3
    addvariable!(p, collect(0.1:0.1:0.5))  # 5
    addcost!(p, WDynRes([0.1, 0.2, 0.3, 0.4], 1, 2)) # m = 4, n = 5
    addcost!(p, WDynRes([0.5], 2, 3))                # m = 1, n = 8
    addcost!(p, WDynRes([0.1, -0.1], 1, 3))          # m = 2, n = 7
    return p
end

wtuple(w) = (w.cost, w.tall, w.normal)
