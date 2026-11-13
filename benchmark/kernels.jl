# P1: does the @turbo kernel earn its place? Times the two dynamic Gram kernels of src/gramkernels.jl against each other.
#
# Regenerate the table (on any machine, including x86-64; needs the sibling checkouts that NLLSModel's Project.toml names
# under [sources], i.e. BlockLinearSystems.jl, FastNorm.jl, OptimizationVariableInterface.jl and SIRBSDispatch.jl beside this repo):
#
#     cd NLLSModel.jl
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark -t 4 benchmark/kernels.jl            # about 10 minutes; writes benchmark/results/<cpu>.md
#
# Environment: BENCH_SECONDS (time per kernel cell, default 0.1), BENCH_QUICK=1 (a small grid, for checking the script).
# The machine is idle-sensitive: close other work first. Nothing here is asserted.
#
# Part 1 times, for identical Vector/Matrix inputs, gram_plain! against gram_turbo! (with and without the second-order
# term, c2 == 0 and c2 != 0) and jtr_plain! against jtr_turbo!, over rows M x columns n, in Float64 and Float32. The ratio
# per cell is time(plain) / time(turbo): above 1 means @turbo is faster. "plain" is the kernel as it is now (with
# @fastmath); the plain kernel before @fastmath was added (F2) is timed too, and the n == 1 column is shown before routing
# (turbo, which is what ran) and after (plain).
# Part 2 times one whole linearization (fdf!, the gradient and Hessian of all costs) with each kernel forced, on 1 and 4 threads.
using BenchmarkTools, NLLSModel, Static, StaticArrays, Random, Printf, Dates

const BENCH_SECONDS = parse(Float64, get(ENV, "BENCH_SECONDS", "0.1"))
const QUICK = get(ENV, "BENCH_QUICK", "0") == "1"
const ROWS = QUICK ? [1, 8, 40] : [1, 2, 3, 8, 17, 23, 40, 100, 1000]
const COLS = QUICK ? [1, 10, 100] : [1, 6, 10, 30, 100, 250]

# ---- The machine ----
function cpuname()
    if Sys.islinux() && isfile("/proc/cpuinfo")
        for line in eachline("/proc/cpuinfo")
            startswith(line, "model name") && return strip(split(line, ':'; limit=2)[2])
        end
    end
    return strip(Sys.cpu_info()[1].model)
end
const CPU = cpuname()
const ARCH = string(Sys.ARCH, ", LLVM CPU target ", Sys.CPU_NAME)

# ---- Part 1: the kernels ----
const K = NLLSModel

# The plain kernels as they were before F2 added @fastmath, kept here so the change can be timed
function jtr_plain_old!(g, J, r)
    M, n = size(J)
    @inbounds for i in 1:n
        acc = zero(eltype(g))
        @simd for k in 1:M
            acc += J[k, i] * r[k]
        end
        g[i] = acc
    end
end
function gram_plain_old!(H, g, J, r, dc, c2)
    M, n = size(J)
    jtr_plain_old!(g, J, r)
    @inbounds for i in 1:n
        for j in 1:i
            acc = zero(eltype(H))
            @simd for k in 1:M
                acc += J[k, i] * J[k, j]
            end
            H[i, j] = acc
        end
    end
    K.gram_weight_mirror!(H, g, n, dc, c2)
    K.gram_scale!(g, n, dc)
end

# Each cell is built by a function, so that the arguments are interpolated into the benchmark
function cell_gram(T, M, n, c2)
    J = randn(T, M, n); r = randn(T, M); H = zeros(T, n, n); g = zeros(T, n)
    dc = T(0.7); c2T = T(c2)
    tp = @belapsed K.gram_plain!($H, $g, $J, $r, $dc, $c2T) seconds=BENCH_SECONDS
    tt = @belapsed K.gram_turbo!($H, $g, $J, $r, $dc, $c2T) seconds=BENCH_SECONDS
    to = @belapsed gram_plain_old!($H, $g, $J, $r, $dc, $c2T) seconds=BENCH_SECONDS
    return tp, tt, to
end
function cell_jtr(T, M, n)
    J = randn(T, M, n); r = randn(T, M); g = zeros(T, n)
    tp = @belapsed K.jtr_plain!($g, $J, $r) seconds=BENCH_SECONDS
    tt = @belapsed K.jtr_turbo!($g, $J, $r) seconds=BENCH_SECONDS
    to = @belapsed jtr_plain_old!($g, $J, $r) seconds=BENCH_SECONDS
    return tp, tt, to
end

function part1()
    results = Dict{Tuple{Symbol, DataType}, Matrix{NTuple{3, Float64}}}()
    for T in (Float64, Float32)
        for (name, f) in ((:gram_c2_zero, (M, n) -> cell_gram(T, M, n, 0.0)), (:gram_c2_nonzero, (M, n) -> cell_gram(T, M, n, 0.3)), (:jtr, (M, n) -> cell_jtr(T, M, n)))
            tab = Matrix{NTuple{3, Float64}}(undef, length(ROWS), length(COLS))
            for (i, M) in enumerate(ROWS), (j, n) in enumerate(COLS)
                tab[i, j] = f(M, n)
            end
            results[(name, T)] = tab
            @info "done" name T
        end
    end
    return results
end

# ---- Part 2: a whole linearization with each kernel forced ----
# The kernel choice is made by `turbo_ok`. Forcing the plain kernel replaces its one specific method; restoring puts the
# original definition back (copied from src/gramkernels.jl; the check below fails loudly if that file has changed).
const PROBE = (Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64})
function setturbo!(on::Bool)
    Core.eval(K, :(turbo_ok(::Type{<:StridedGram{T, 2}}, ::Type{<:StridedGram{T, 1}}, ::Type{<:StridedGram{T, 2}}, ::Type{<:StridedGram{T, 1}}) where {T <: GramFloat} = $on))
    Base.invokelatest(K.turbo_ok, PROBE...) == on || error("could not force the kernel: src/gramkernels.jl has changed")
end

# Dynamic variable, residual length per cost either run-time (D == 0) or static
struct ProcRes{D} <: NLLSModel.AbstractResidual
    a::Vector{Float64}
    b::Vector{Float64}
end
Base.eltype(::ProcRes) = Float64
NLLSModel.ndeps(::ProcRes) = static(1)
NLLSModel.nres(r::ProcRes{D}) where D = D == 0 ? length(r.a) : static(D)
NLLSModel.varindices(::ProcRes) = 1
NLLSModel.getvars(::ProcRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
function rotate(x, b, d)
    T = eltype(x)
    W = zeros(T, d, d); k = 1
    for j in 1:d, i in j+1:d
        W[i, j] = x[k]; W[j, i] = -x[k]; k += 1
    end
    return b .+ W * b .+ 0.5 .* (W * (W * b))
end
NLLSModel.computeresidual(r::ProcRes, x) = r.a .- rotate(x, r.b, length(r.b))

function procrustes(d, npts, staticres)
    rng = MersenneTwister(1)
    R = ProcRes{staticres ? d : 0}
    problem = Model(Vector{Float64}, R)
    addvariable!(problem, 0.01 .* randn(rng, d * (d - 1) ÷ 2))
    for _ in 1:npts
        b = randn(rng, d)
        addcost!(problem, R(b .+ 0.05 .* randn(rng, d), b))
    end
    return problem
end

# A linear residual A x - b with a hand-written Jacobian: the kernels dominate (no autodiff)
struct LinRes <: NLLSModel.AbstractResidual
    A::Matrix{Float64}
    b::Vector{Float64}
end
Base.eltype(::LinRes) = Float64
NLLSModel.ndeps(::LinRes) = static(1)
NLLSModel.nres(r::LinRes) = length(r.b)
NLLSModel.varindices(::LinRes) = 1
NLLSModel.getvars(::LinRes, vars::AbstractVector) = (vars[1]::Vector{Float64},)
NLLSModel.computeresidual(r::LinRes, x) = r.A * x .- r.b
NLLSModel.computeresjac(varflags, r::LinRes, x) = (r.A * x .- r.b, r.A)

function linear(M, n, ncosts)
    rng = MersenneTwister(2)
    problem = Model(Vector{Float64}, LinRes)
    addvariable!(problem, randn(rng, n))
    for _ in 1:ncosts
        addcost!(problem, LinRes(randn(rng, M, n), randn(rng, M)))
    end
    return problem
end

function linearization_time(problem, nthreads)
    f, fdf!, x, H = optimvars(problem, nothing, static(nthreads))
    n = length(x)
    g = zeros(n); Hm = H === nothing ? zeros(n, n) : H
    fdf!(g, Hm, x)
    return @belapsed($fdf!($g, $Hm, $x); samples=15, evals=1, seconds=60) # Interpolated; the minimum over 15 runs
end

const WHOLE = QUICK ?
    [("Procrustes d=8, 400 points, static nres 8 (static-column kernel: not affected)", () -> procrustes(8, 400, true)),
     ("linear, M=100, n=30, 100 costs, hand-written Jacobian", () -> linear(100, 30, 100))] :
    [("Procrustes d=12, 4000 points, run-time nres", () -> procrustes(12, 4000, false)),
     ("Procrustes d=24, 2000 points, run-time nres", () -> procrustes(24, 2000, false)),
     ("Procrustes d=24, 2000 points, static nres 24 (above MAX_STATIC_ROWS)", () -> procrustes(24, 2000, true)),
     ("linear, M=100, n=30, 1000 costs, hand-written Jacobian", () -> linear(100, 30, 1000)),
     ("linear, M=1000, n=30, 200 costs, hand-written Jacobian", () -> linear(1000, 30, 200)),
     ("linear, M=40, n=100, 500 costs, hand-written Jacobian", () -> linear(40, 100, 500))]

function part2()
    out = Tuple{String, Int, Float64, Float64}[]
    for (name, make) in WHOLE, nt in (1, 4)
        problem = make()
        setturbo!(true);  tt = Base.invokelatest(linearization_time, problem, nt)
        setturbo!(false); tp = Base.invokelatest(linearization_time, problem, nt)
        setturbo!(true)
        push!(out, (name, nt, tp, tt))
        @info "done" name nt tp tt
    end
    return out
end

# ---- Report ----
function report(io, results, whole)
    println(io, "# @turbo against plain kernels (P1)\n")
    println(io, "- CPU: ", CPU)
    println(io, "- Architecture: ", ARCH)
    println(io, "- Julia ", VERSION, ", ", Threads.nthreads(), " threads, ", Sys.CPU_THREADS, " logical cores; run ", Dates.format(now(), "yyyy-mm-dd"))
    println(io, "- Seconds per cell: ", BENCH_SECONDS, if QUICK; " (QUICK grid: not for decisions)" else "" end)
    println(io, "\nEach cell is `time(plain) / time(turbo)`: above 1, `@turbo` is faster; below 1, the plain `@simd` kernel is.\n")
    for T in (Float64, Float32), (name, title) in ((:gram_c2_zero, "gram, c2 == 0"), (:gram_c2_nonzero, "gram, c2 != 0"), (:jtr, "jtr (J'r only)"))
        tab = results[(name, T)]
        println(io, "## $title, $T\n")
        println(io, "| M \\ n | ", join(COLS, " | "), " |")
        println(io, "|---|", join(fill("---:", length(COLS)), "|"), "|")
        for (i, M) in enumerate(ROWS)
            println(io, "| $M | ", join((@sprintf("%.2f", tab[i, j][1] / tab[i, j][2]) for j in eachindex(COLS)), " | "), " |")
        end
        rat = [t[1] / t[2] for t in tab]
        println(io, "\nrange ", @sprintf("%.2f–%.2f", minimum(rat), maximum(rat)), ", geometric mean ", @sprintf("%.2f", exp(sum(log, rat) / length(rat))), "; cells within 10%: ", count(r -> 0.9 <= r <= 1.1, rat), " of ", length(rat), "\n")
    end
    println(io, "## Plain kernel before and after `@fastmath` (F2)\n")
    println(io, "Each cell is `time(plain without @fastmath) / time(plain with @fastmath)`: above 1, `@fastmath` is faster; below 1 it is slower.\n")
    for T in (Float64, Float32), (name, title) in ((:gram_c2_zero, "gram"), (:jtr, "jtr"))
        tab = results[(name, T)]
        println(io, "### $title, $T\n")
        println(io, "| M \\ n | ", join(COLS, " | "), " |")
        println(io, "|---|", join(fill("---:", length(COLS)), "|"), "|")
        for (i, M) in enumerate(ROWS)
            println(io, "| $M | ", join((@sprintf("%.2f", tab[i, j][3] / tab[i, j][1]) for j in eachindex(COLS)), " | "), " |")
        end
        rat = [t[3] / t[1] for t in tab]
        println(io, "\nrange ", @sprintf("%.2f–%.2f", minimum(rat), maximum(rat)), ", geometric mean ", @sprintf("%.2f", exp(sum(log, rat) / length(rat))), "; cells slower with `@fastmath` by more than 10%: ", count(r -> r < 0.9, rat), " of ", length(rat), "\n")
    end
    println(io, "## The n == 1 column, before and after routing to the plain kernel (F2)\n")
    println(io, "Before: `@turbo` (what ran). After: plain with `@fastmath` (what runs). Times in ns; the last column is before / after.\n")
    println(io, "| kernel | type | M | before (turbo) | after (plain) | before / after |")
    println(io, "|---|---|---:|---:|---:|---:|")
    for T in (Float64, Float32), (name, title) in ((:gram_c2_zero, "gram"), (:jtr, "jtr")), (i, M) in enumerate(ROWS)
        j = findfirst(==(1), COLS)
        isnothing(j) && continue
        tp, tt, _ = results[(name, T)][i, j]
        println(io, "| $title | $T | $M | ", @sprintf("%.1f", 1e9 * tt), " | ", @sprintf("%.1f", 1e9 * tp), " | ", @sprintf("%.2f", tt / tp), " |")
    end
    println(io)
    println(io, "## Whole linearization (fdf! over all costs), kernel forced\n")
    println(io, "| Problem | threads | plain (ms) | turbo (ms) | plain / turbo |")
    println(io, "|---|---:|---:|---:|---:|")
    for (name, nt, tp, tt) in whole
        println(io, "| $name | $nt | ", @sprintf("%.3f", 1e3 * tp), " | ", @sprintf("%.3f", 1e3 * tt), " | ", @sprintf("%.2f", tp / tt), " |")
    end
    println(io, "\n## Raw times of Part 1 (ns), `plain/turbo/plain without @fastmath`\n")
    for T in (Float64, Float32), name in (:gram_c2_zero, :gram_c2_nonzero, :jtr)
        tab = results[(name, T)]
        println(io, "### $name, $T\n\n| M \\ n | ", join(COLS, " | "), " |\n|---|", join(fill("---:", length(COLS)), "|"), "|")
        for (i, M) in enumerate(ROWS)
            println(io, "| $M | ", join((@sprintf("%.0f/%.0f/%.0f", 1e9 * tab[i, j][1], 1e9 * tab[i, j][2], 1e9 * tab[i, j][3]) for j in eachindex(COLS)), " | "), " |")
        end
        println(io)
    end
end

function main()
    @info "CPU" CPU ARCH
    setturbo!(true)
    results = part1()
    whole = part2()
    report(stdout, results, whole)
    dir = joinpath(@__DIR__, "results")
    mkpath(dir)
    file = joinpath(dir, replace(CPU, r"[^A-Za-z0-9]+" => "_") * (QUICK ? "_quick" : "") * ".md")
    open(io -> report(io, results, whole), file, "w")
    @info "written" file
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
