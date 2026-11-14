# W1: calibrate the three threading floors (src/multithreading.jl) in the units of their estimates.
#
#     cd NLLSModel.jl
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark -t 8 benchmark/threading.jl         # writes benchmark/results/threading_<cpu>.md
#
# For each problem family, thread count and kind of work (cost evaluation f, tall fill resjac!, normal-equations
# linearization fdf!), it times the serial and the threaded evaluation (forced with a floor of 10^12 and of 0) over
# problem sizes spaced by a factor of about 1.6 in the estimate of that work, and reports the break-even: the smallest
# estimate from which the threaded evaluation is faster at every larger size measured. Times are minima over repeated
# evaluations. The machine is idle-sensitive: close other work first. Environment: BENCH_THREADS (default "2,4,8").
using NLLSModel, Static, StaticArrays, Printf, Dates

# ---- families ----

# Bundle adjustment: 2 rows on a 9-dimensional camera and a 3-dimensional point (12 columns)
struct BARes <: NLLSModel.AbstractResidual
    meas::SVector{2, Float64}
    cam::Int
    pt::Int
end
NLLSModel.ndeps(::BARes) = static(2)
NLLSModel.nres(::BARes) = static(2)
NLLSModel.varindices(r::BARes) = SVector(r.cam, r.pt)
NLLSModel.getvars(r::BARes, vars::AbstractVector) = (vars[r.cam]::SVector{9, Float64}, vars[r.pt]::SVector{3, Float64})
function NLLSModel.computeresidual(r::BARes, c, p)
    q = SVector(c[1] * p[1] + c[2] * p[2] + c[3] * p[3] + c[7], c[4] * p[1] + c[5] * p[2] + c[6] * p[3] + c[8], p[3] + c[9])
    return SVector(q[1] / q[3], q[2] / q[3]) - r.meas
end
Base.eltype(::BARes) = Float64
function bundle(ncosts)
    ncams = max(2, round(Int, sqrt(ncosts / 50)))
    npts = cld(ncosts, ncams)
    p = Model(Union{SVector{9, Float64}, SVector{3, Float64}}, BARes)
    for i in 1:ncams
        addvariable!(p, SVector(1.0, 0.01i, 0.0, 0.0, 1.0, 0.02i, 0.1i, -0.1i, 5.0))
    end
    for j in 1:npts
        addvariable!(p, SVector(sin(j), cos(j), 0.1 * sin(3j)))
    end
    k = 0
    for j in 1:npts, i in 1:ncams
        (k += 1) > ncosts && break
        addcost!(p, BARes(SVector(0.01 * sin(i + j), 0.01 * cos(i * j)), i, ncams + j))
    end
    return p
end

# Procrustes: d rows on one rotation of d(d-1)/2 parameters (a dynamic variable)
struct ProcRes{D} <: NLLSModel.AbstractResidual
    a::Vector{Float64}
    b::Vector{Float64}
end
NLLSModel.ndeps(::ProcRes) = static(1)
NLLSModel.nres(::ProcRes{D}) where D = static(D)
NLLSModel.varindices(::ProcRes) = 1
NLLSModel.getvars(::ProcRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
function rotate(x, b, d)
    W = zeros(eltype(x), d, d); k = 1
    for j in 1:d, i in j+1:d
        W[i, j] = x[k]; W[j, i] = -x[k]; k += 1
    end
    return b .+ W * b .+ 0.5 .* (W * (W * b))
end
NLLSModel.computeresidual(r::ProcRes{D}, x) where D = r.a .- rotate(x, r.b, D)
Base.eltype(::ProcRes) = Float64
function procrustes(d, ncosts)
    p = Model(Vector{Float64}, ProcRes{d})
    addvariable!(p, [0.01 * sin(k) for k in 1:d * (d - 1) ÷ 2])
    for k in 1:ncosts
        b = [cos(k + i) for i in 1:d]
        addcost!(p, ProcRes{d}(b .+ 0.01 .* sin.(1:d), b))
    end
    return p
end

# Curve fit: one scalar residual per point on one dynamic 5-parameter variable
struct CurveRes <: NLLSModel.AbstractResidual
    t::Float64
    y::Float64
end
NLLSModel.ndeps(::CurveRes) = static(1)
NLLSModel.nres(::CurveRes) = static(1)
NLLSModel.varindices(::CurveRes) = 1
NLLSModel.getvars(::CurveRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(r::CurveRes, a) = SVector(a[1] * exp(-a[2] * r.t) + a[3] * sin(a[4] * r.t + a[5]) - r.y)
Base.eltype(::CurveRes) = Float64
function curvefit(ncosts)
    p = Model(Vector{Float64}, CurveRes)
    addvariable!(p, [1.0, 0.5, 0.3, 2.0, 0.1])
    for k in 1:ncosts
        t = 4k / ncosts
        addcost!(p, CurveRes(t, exp(-0.4t) + 0.2 * sin(2.1t)))
    end
    return p
end

# Pose chain: 6 rows between consecutive 6-dimensional poses (12 columns)
struct PoseRes <: NLLSModel.AbstractResidual
    meas::SVector{6, Float64}
    i::Int
end
NLLSModel.ndeps(::PoseRes) = static(2)
NLLSModel.nres(::PoseRes) = static(6)
NLLSModel.varindices(r::PoseRes) = SVector(r.i, r.i + 1)
NLLSModel.getvars(r::PoseRes, vars::AbstractVector) = (vars[r.i]::SVector{6, Float64}, vars[r.i + 1]::SVector{6, Float64})
function NLLSModel.computeresidual(r::PoseRes, a, b)
    ca, sa = cos(a[3]), sin(a[3])
    return SVector(ca * (b[1] - a[1]) + sa * (b[2] - a[2]), ca * (b[2] - a[2]) - sa * (b[1] - a[1]), b[3] - a[3],
                   b[4] - a[4] * b[6], b[5] - a[5] + 0.1 * a[4]^2, b[6] - a[6]) - r.meas
end
Base.eltype(::PoseRes) = Float64
function posechain(ncosts)
    p = Model(SVector{6, Float64}, PoseRes)
    for i in 1:ncosts + 1
        addvariable!(p, SVector(0.1i, 0.05 * sin(i), 0.01i, 0.0, 0.0, 0.1))
    end
    for i in 1:ncosts
        addcost!(p, PoseRes(SVector(0.1, 0.0, 0.01, 0.0, 0.0, 0.0), i))
    end
    return p
end

const FAMILIES = [
    ("bundle adjustment (2 x 12)", bundle),
    ("Procrustes d=5 (5 x 10)", n -> procrustes(5, n)),
    ("Procrustes d=10 (10 x 45)", n -> procrustes(10, n)),
    ("Procrustes d=20 (20 x 190)", n -> procrustes(20, n)),
    ("curve fit, dynamic (1 x 5)", curvefit),
    ("pose chain (6 x 12)", posechain),
]

# ---- timing ----

function mintime(f; budget=0.05, maxreps=2000)
    f(); f() # warm
    best = Inf; t0 = time(); reps = 0
    while reps < 3 || (time() - t0 < budget && reps < maxreps)
        t = @elapsed f()
        best = min(best, t); reps += 1
    end
    return best
end

const HUGE = 10^12

# Evaluators for one kind of work, forced serial (floor HUGE) or threaded (floor 0) on T threads
function evaluator(kind, p, T, floor)
    if kind === :cost
        f, _, x, _ = optimvars(p, nothing, static(T); min_work=HUGE, min_work_cost=floor)
        return () -> f(x)
    elseif kind === :linearization
        _, fdf!, x, H = optimvars(p, nothing, static(T); min_work=floor, min_work_cost=HUGE)
        n = Int(sum(v -> dynamic(NLLSModel.nvars(v)), p.variables))
        g = zeros(n); Hb = H === nothing ? zeros(n, n) : H
        return () -> fdf!(g, Hb, x)
    else
        _, _, resjac!, x, J = optimvars_nls(p; numthreads=static(T), min_work_per_thread=floor)
        r = zeros(resjac!.layout.M)
        return () -> resjac!(r, J, x)
    end
end

estimate(kind, p) = (w = NLLSModel.workestimates(p, nothing); kind === :cost ? w.cost : kind === :tall ? w.tall : w.linearization)

# Sizes spanning the estimate from lo to hi; the largest size is also capped so a dense Hessian or Jacobian stays small
function sweep(kind, make, T; lo=parse(Int, get(ENV, "BENCH_LO", "20")), hi=parse(Int, get(ENV, "BENCH_HI", "3000000")))
    per = estimate(kind, make(64)) / 64
    rows = Tuple{Int, Int, Float64, Float64}[]
    W = lo
    while W <= hi
        ncosts = max(1, round(Int, W / per))
        if !isempty(rows) && ncosts <= rows[end][1] # the same size again
            W = ceil(Int, W * 1.6)
            continue
        end
        p = make(ncosts)
        w = estimate(kind, p)
        n = Int(sum(v -> dynamic(NLLSModel.nvars(v)), p.variables))
        (kind === :linearization && n > 3000 && length(p.variables) < 4) && break # a dense single-variable Hessian
        (kind === :tall && 8 * NLLSModel.nresiduals(p) * n > 2^28) && break
        ts = mintime(evaluator(kind, p, T, HUGE))
        tt = mintime(evaluator(kind, p, T, 0))
        push!(rows, (ncosts, w, ts, tt))
        W = ceil(Int, W * 1.6)
    end
    return rows
end

# The break-even as an interval (lo, hi]: lo the largest estimate at which threaded was not faster (0 if it always was),
# hi the next size measured (Inf if threaded never won from some size on). Threads split the costs, so the smallest
# size with more than one cost bounds hi from above.
function breakeven(rows)
    idx = findlast(r -> r[4] >= r[3] && r[1] > 1, rows) # one cost cannot be split, so it never counts as a loss
    idx === nothing && return (0.0, Float64(rows[1][2]))
    idx == length(rows) && return (Float64(rows[idx][2]), Inf)
    return (Float64(rows[idx][2]), Float64(rows[idx + 1][2]))
end
fmtint(x) = isinf(x) ? "∞" : string(round(Int, x))
cell((lo, hi)) = "(" * fmtint(lo) * ", " * fmtint(hi) * "]"

# Floors F within a factor of two of every break-even: lo / 2 <= F <= 2 hi for each interval
function fitrange(intervals)
    a = maximum(first, intervals) / 2
    b = minimum(last, intervals) * 2
    return a, b
end

function main()
    threads = parse.(Int, split(get(ENV, "BENCH_THREADS", "2,4,8"), ","))
    filter!(T -> T <= Threads.nthreads(), threads)
    cpu = strip(Sys.cpu_info()[1].model)
    out = IOBuffer()
    println(out, "# Threading floors, W1 calibration\n")
    println(out, "Machine: ", cpu, ", ", Sys.CPU_THREADS, " logical CPUs; Julia ", VERSION, ", -t ", Threads.nthreads(), "; ", Dates.format(now(), "yyyy-mm-dd"), ".\n")
    println(out, "Break-even as an interval (lo, hi]: lo the largest estimate at which the threaded evaluation was not faster (0 if it always was), hi the next size (∞ if it never won from some size on). A floor fits a family within a factor of two if lo/2 ≤ F ≤ 2·hi. A size with one cost is never counted as a loss: threads split the costs, so one cost cannot gain whatever the floor.\n")
    summary = Dict{Tuple{Symbol, Int}, Vector{Any}}()
    details = IOBuffer()
    for kind in (:cost, :tall, :linearization), (name, make) in FAMILIES, T in threads
        rows = sweep(kind, make, T)
        iv = breakeven(rows)
        push!(get!(summary, (kind, T), Any[]), iv)
        println(details, "\n### ", kind, ", ", name, ", ", T, " threads\n")
        println(details, "| costs | estimate | serial (µs) | threaded (µs) | speed-up |\n|---:|---:|---:|---:|---:|")
        for (nc, w, ts, tt) in rows
            @printf(details, "| %d | %d | %.2f | %.2f | %.2f |\n", nc, w, 1e6ts, 1e6tt, ts / tt)
        end
        @printf(stderr, "%-14s %-28s T=%d break-even %s\n", kind, name, T, cell(iv))
    end
    for kind in (:cost, :tall, :linearization)
        println(out, "\n## ", kind, "\n")
        println(out, "| family | ", join(("$T threads" for T in threads), " | "), " |")
        println(out, "|---|", join(("---:" for _ in threads), "|"), "|")
        for (i, (name, _)) in enumerate(FAMILIES)
            cells = map(T -> cell(summary[(kind, T)][i]), threads)
            println(out, "| ", name, " | ", join(cells, " | "), " |")
        end
        ranges = map(T -> fitrange(summary[(kind, T)]), threads)
        println(out, "| **floors within 2× of all** | ", join((a <= b ? "[" * fmtint(a) * ", " * fmtint(b) * "]" : "none (" * fmtint(a) * " > " * fmtint(b) * ")" for (a, b) in ranges), " | "), " |")
        a, b = fitrange(reduce(vcat, (summary[(kind, T)] for T in threads)))
        println(out, "\nAll thread counts together: ", a <= b ? "[" * fmtint(a) * ", " * fmtint(b) * "]" : "none (" * fmtint(a) * " > " * fmtint(b) * ")", ".")
    end
    println(out, "\n## Measurements\n")
    print(out, String(take!(details)))
    file = joinpath(@__DIR__, "results", "threading_" * replace(cpu, r"[^A-Za-z0-9]+" => "_") * ".md")
    write(file, String(take!(out)))
    println(stderr, "wrote ", file)
end

main()
