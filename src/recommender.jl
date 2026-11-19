# The optimizer recommender: `recommendoptimizer(model)` chooses a formulation, reduction, storage, linear solver and iterator
# by hand-written rules, from the model's structure and from one linearization and factorization of its Hessian system at
# the model's current values (improvements.md item 25), and, when the Hessian is ill-conditioned or does not factorize
# undamped, from one evaluation of the Jacobian, to test its rank. The rules are pure functions of the features, below, so
# that they can be tested without a model and without timing noise.

# The thresholds of the rules. Every one is a starting value, not measured (rules.md 26); they are listed for measurement
# in improvements.md item 25.
const REC_JACOBIAN_CONDITION = 1e12       # F: the Jacobian path at or above this condition estimate. Starting value, not measured.
const REC_DOGLEG_CONDITION = 1e8      # I: dogleg only below this condition estimate. Starting value, not measured.
const REC_DIAGSPREAD = 1e6            # I: dogleg only below, and D'D-scaled damping at or above, this spread of the
                                      # Hessian's diagonal. Starting value, not measured.
const REC_DOGLEG_SOLVE_RATIO = 1.0    # I: dogleg only when a solve costs at least this many linearizations. Starting
                                      # value, not measured.
const REC_INVERSE_ITERATIONS = 2      # Steps of inverse iteration in the condition estimate, and for the rank test's
                                      # direction. Starting value, not measured.
const REC_RANK_TOLERANCE = 1e-12      # F: the column-scaled Jacobian is numerically rank-deficient when its norm along the
                                      # weakest direction of the scaled Hessian is below this. Starting value, not measured.
const REC_RANK_DAMPINGS = (1e-14, 1e-12, 1e-10) # F: the scaled dampings tried, in order, when the undamped Hessian does
                                      # not factorize, to find the rank test's direction. Starting values, not measured.
const REC_RANK_REFINEMENTS = 1        # F: refinement steps of that direction with products of the Jacobian. Starting
                                      # value, not measured.

# What the recommender measured: the structure, and the probe of the Hessian system at the model's current values.
# Times are in seconds.
struct RecommenderFeatures
    nvariables::Int           # Unfixed variables
    dimension::Int            # Their total dimension, n
    ncosts::Int               # Costs, of every kind
    nresiduals::Int           # Residual rows, m, of the residual costs
    allresiduals::Bool        # Every cost is an AbstractResidual
    hessiansystem::Symbol     # The Hessian path's system: :static, :dense, :sparse, :schur_dense or :schur_sparse
    retained::Int             # Under elimination on the Hessian path: the retained dimension (0 without)
    eliminated::Int           # ... and the eliminated dimension (0 without)
    diag_min::Float64         # Smallest entry of the Hessian's diagonal
    diag_max::Float64         # Largest
    diagspread::Float64       # diag_max / diag_min, Inf when diag_min is not positive
    status::SolveStatus       # Of the undamped solve
    condition::Float64        # Lower bound on the condition number of the Jacobi-scaled Hessian, Inf if not factorizable
    ranktest::Symbol          # The rank test of the Jacobian: :notneeded, :ran, or why it could not run: :nodiagonal (a
                              # non-positive diagonal entry), :nofactorization (no damping tried factorizes), :nojacobian
                              # (the Jacobian path cannot be built for the model)
    rank_damping::Float64     # The scaled damping of the factorization the test used (0 undamped); NaN if it did not run
    rank_t::Float64           # ‖J D^-½ w‖ along the weakest direction w of the scaled Hessian; NaN if it did not run
    t_cost::Float64           # One cost evaluation
    t_linearize::Float64      # One linearization (zero! and fdf!)
    t_solve::Float64          # One undamped solve
end

# The recommendation: the five choices, the keywords it was made with, the evidence, and the rule that fired for each
# choice. Documented in docstrings.jl.
struct OptimizationRecommendation{U, N}
    formulation::Symbol
    reduction::Symbol
    storage::Symbol
    linsolver::Symbol
    iterator::Symbol
    unfixed::U                # As given to recommendoptimizer
    numthreads::N             # As given
    max_dense_bytes::Int
    features::RecommenderFeatures
    rules::NamedTuple{(:formulation, :reduction, :storage, :linsolver, :iterator), NTuple{5, Symbol}} # The rule that fired for each
    fallback::String          # Why the Jacobian path could not be built, when F's fallback fired; otherwise empty
end

# The Hessian path's system, from the prototype `optimvars` returned with `reduction = storage = :auto`
hessiansystemkind(::Nothing) = :dense # Several variables, dense: the optimizer builds the system from n
hessiansystemkind(::StaticMatrix) = :static
hessiansystemkind(::Matrix) = :dense
hessiansystemkind(::SparseMatrixCSC) = :sparse
hessiansystemkind(::SchurBlockHessian{T, <:Matrix}) where T = :schur_dense
hessiansystemkind(::SchurBlockHessian) = :schur_sparse

# The `reduction` and `storage` that build a system of each kind
reductionstorage(kind::Symbol) = kind === :schur_dense ? (:eliminate, :dense) : kind === :schur_sparse ? (:eliminate, :sparse) :
    kind === :sparse ? (:none, :sparse) : (:none, :dense)

# The same, read from the Jacobian prototype `optimvars_nls` returned with `:auto`
jacobianreductionstorage(::Matrix) = (:none, :dense)
jacobianreductionstorage(::SparseMatrixCSC) = (:none, :sparse)
jacobianreductionstorage(::SchurBlockJacobian{T, <:Matrix}) where T = (:eliminate, :dense)
jacobianreductionstorage(::SchurBlockJacobian) = (:eliminate, :sparse)

# Retained and eliminated dimensions under elimination on the Hessian path
function elimdims(H::SchurBlockHessian)
    sizes = BlockLinearSystems.blocksizes(H.structure)
    nr = nretained(H.structure)
    return (sum(view(sizes, 1:nr); init=0), sum(view(sizes, nr+1:length(sizes)); init=0))
end
elimdims(H) = (0, 0)

# Rule L, the linear solver of the Hessian formulation: the explicit solver that `linsolver = :auto` resolves to for the
# system (Cholesky for a static, dense or Schur-complement system, LDLᵀ for a sparse one; DampedLinearSolve's `solve!` and
# BlockLinearSystems' Schur system), but `:ldl` whenever some cost is not a residual, since its Hessian can be indefinite.
# Only solvers that are always available: NLLSModel cannot see whether Krylov.jl or Pardiso.jl is loaded.
function rule_linsolver(allresiduals::Bool, hessiansystem::Symbol)
    allresiduals || return (:ldl, :L_nonresidual)
    hessiansystem === :sparse && return (:ldl, :L_sparse)
    hessiansystem === :static && return (:cholesky, :L_static)
    hessiansystem === :dense && return (:cholesky, :L_dense)
    return (:cholesky, :L_schur)
end

# Whether rule F needs the rank test of the Jacobian: every cost is a residual, and the Hessian is ill-conditioned or
# could not be factorized undamped. The one predicate says both when the test runs and when the rule reads it.
needsranktest(allresiduals::Bool, status::SolveStatus, condition::Real) =
    allresiduals && !(status == Success && condition < REC_JACOBIAN_CONDITION)
needsranktest(f::RecommenderFeatures) = needsranktest(f.allresiduals, f.status, f.condition)

# Rule F, the formulation: the Hessian path when the scaled Hessian is well conditioned. Otherwise the rank test of the
# Jacobian decides: a numerically rank-deficient J (a gauge freedom) stays on the Hessian path, since factorizing J gains
# nothing and damping handles it; a full-rank one goes to the Jacobian path, since forming JᵀJ squares its conditioning.
# Where the test could not run, the Hessian path.
function rule_formulation(f::RecommenderFeatures)
    f.allresiduals || return (:hessian, :F_nonresidual)
    needsranktest(f) || return (:hessian, :F_wellconditioned)
    f.ranktest === :nojacobian && return (:hessian, :F_jacobian_unavailable)
    f.ranktest === :ran || return (:hessian, :F_singular)
    f.rank_t < REC_RANK_TOLERANCE && return (:hessian, :F_rankdeficient)
    return (:jacobian, :F_illconditioned)
end

# The tests of dogleg's rule, each with whether it passed
dogleg_tests(f::RecommenderFeatures) = (
    success = f.status == Success,
    condition = f.condition < REC_DOGLEG_CONDITION,
    diagspread = f.diagspread < REC_DIAGSPREAD,
    solvecost = f.t_solve >= REC_DOGLEG_SOLVE_RATIO * f.t_linearize,
)

# Rule I, the iterator: dogleg (which does not solve again after a rejected step, and has no scaling of its own and no
# damping) when the solve is the expensive part and the system is well scaled and well conditioned; D'D-scaled
# Levenberg-Marquardt when the diagonal is badly spread; otherwise Levenberg-Marquardt.
function rule_iterator(f::RecommenderFeatures)
    all(dogleg_tests(f)) && return (:dogleg, :I_dogleg)
    f.diagspread >= REC_DIAGSPREAD && return (:levenbergmarquardt_dtd, :I_dtd)
    return (:levenbergmarquardt, :I_lm)
end

# A fixed unit vector to start inverse iteration from: deterministic, and with no special direction a structured
# matrix is likely to share (alternating signs, decreasing magnitudes)
function inverse_iteration_start(n::Int)
    w = [(isodd(i) ? 1.0 : -1.0) / sqrt(i) for i in 1:n]
    return w ./ norm(w)
end

# Inverse iteration on the Jacobi-scaled matrix D^-½ (A + λD) D^-½ (D the diagonal of A, λ the damping set on the
# system), which needs only solves of the system: b = D^½ w, solve, y = D^½ x, ρ = ‖y‖ / ‖w‖, w = y / ‖y‖. Returns ρ and
# the last w, a unit vector that approaches the weakest direction; ρ is Inf (and w the start) if a solve fails. Undamped,
# ρ is the condition estimate: a lower bound on the condition number of D^-½ A D^-½, since that matrix has a unit
# diagonal, so its largest eigenvalue is at least 1, and ρ is at most the largest eigenvalue of its inverse. Overwrites
# the system's b and x.
function scaled_inverse_iteration(ls, linsolver::Symbol, sqrtd::Vector{Float64})
    w = inverse_iteration_start(length(sqrtd))
    rho = Inf
    for _ in 1:REC_INVERSE_ITERATIONS
        DampedLinearSolve.getb(ls) .= sqrtd .* w
        solve!(ls, linsolver) == Success || return (Inf, w)
        y = sqrtd .* DampedLinearSolve.getx(ls)
        rho = norm(y) # ‖w‖ = 1; positive, since b is not zero
        w = y ./ rho
    end
    return (rho, w)
end

# The order of the unfixed variables in each path's system, as indices into `model.variables`: the Hessian path's from its
# `fdf!` (sorted by size, or the elimination's order), the Jacobian path's from its `resjac!` (without elimination, the
# model's order)
varorder(fdf!::SingleVarCostGradHess, model) = [Int(fdf!.unfixed)]
varorder(fdf!::Union{MultiVarDenseFDF, MultiVarSparseFDF, MultiVarSchurFDF}, model) = varorder(fdf!.unfixed_inds, model)
varorder(resjac!::NLSResJac, model) = varorder(resjac!.unfixed_inds, model)
varorder(::Nothing, model) = collect(1:length(model.variables))
varorder(inds::Integer, model) = [Int(inds)]
varorder(inds::AbstractVector{<:Integer}, model) = collect(Int, inds)

# The permutation p of scalar indices: entry i of the Hessian system is column p[i] of the Jacobian
function hessian_to_jacobian(hessorder::Vector{Int}, jacorder::Vector{Int}, model)
    dims = [block_dimension(model.variables[v]) for v in jacorder]
    start = Dict(v => o for (v, o) in zip(jacorder, cumsum([0; dims[1:end-1]])))
    return reduce(vcat, [start[v] .+ (1:block_dimension(model.variables[v])) for v in hessorder]; init=Int[])
end

# The readings ‖J D^-½ w‖ for a unit vector w in the Hessian system's order, before and after each of `refinements`
# refinements of w. A refinement is one step of inverse iteration on the exact JᵀJ, using the factorization held by `ls`
# (of H + λD, with H = JᵀJ up to rounding): g = D^-½ Jᵀ J D^-½ w, w ← w − D^½ (H + λD)⁻¹ D^½ g, then renormalized.
# `solve!` solves (H + λD) x = b, so b = D^½ g is JᵀJ D^-½ w in the Hessian's order, and the step subtracts D^½ x. Why:
# w comes from the rounded H, so it leaks out of J's null space by about (rounding of H) / (smallest non-zero scaled
# singular value of J), and the step removes the leak. On a J of full rank the step is zero but for rounding, so the
# refined w can be noise; but every reading, of a unit vector, is an upper bound on the smallest singular value of the
# column-scaled J, so the rank test reads the smallest. `w` is whatever vector a solve of the system gives (a static one
# for a static system). Overwrites the system's b and x; leaves `w` unchanged.
function rank_readings(J, p::Vector{Int}, sqrtd::Vector{Float64}, w::AbstractVector{Float64}, ls, linsolver::Symbol;
        refinements::Integer=REC_RANK_REFINEMENTS)
    v = zeros(size(J, 2))
    scaledproduct(w) = (v[p] .= w ./ sqrtd; J * v) # J D^-½ w, with w permuted into J's column order
    readings = [norm(scaledproduct(w))]
    for _ in 1:refinements
        DampedLinearSolve.getb(ls) .= view(J' * scaledproduct(w), p)
        solve!(ls, linsolver) # The factorization that gave w, so it succeeds
        w = w .- sqrtd .* DampedLinearSolve.getx(ls)
        w = w ./ norm(w)
        push!(readings, norm(scaledproduct(w)))
    end
    return readings
end

# The Jacobian for the rank test, evaluated once at the model's current values, as a plain matrix (`optimvars_nls` with no
# elimination), and the permutation from the Hessian system's order `hessorder` to its columns; with the prototype
# `optimvars_nls` builds with `:auto`. Without elimination the two prototypes are the same, so it is built once. An
# `ArgumentError` when the Jacobian path cannot be built for the model.
function rank_jacobian(model, hessorder::Vector{Int}; unfixed, numthreads, max_dense_bytes)
    _, _, resjac!, x0, J_auto = optimvars_nls(model; unfixed, numthreads, max_dense_bytes)
    J = J_auto
    J_auto isa SchurBlockJacobian &&
        ((_, _, resjac!, x0, J) = optimvars_nls(model; unfixed, numthreads, max_dense_bytes, reduction=:none))
    J = copy(J)
    resjac!(zeros(size(J, 1)), J, x0)
    return J_auto, J, hessian_to_jacobian(hessorder, varorder(resjac!, model), model)
end

# The rank test of the Jacobian (rule F), when the Hessian is ill-conditioned or could not be factorized undamped. The
# direction w is the weakest of the scaled Hessian, by inverse iteration on the undamped factorization (`w_undamped`,
# from the condition estimate, when `condition` is finite) or on the first scaled damping of `REC_RANK_DAMPINGS` that
# factorizes; then one evaluation of J, at the model's current values, read along w. Returns (ranktest, damping, t, the
# prototype `optimvars_nls` builds with `:auto` or nothing, why the Jacobian path cannot be built or "").
function jacobian_rank_test(model, ls, linsolver::Symbol, d::Vector{Float64}, condition::Float64, w_undamped, hessorder;
        unfixed, numthreads, max_dense_bytes)
    minimum(d) > 0 || return (:nodiagonal, NaN, NaN, nothing, "")
    sqrtd = sqrt.(d)
    damping, w = 0.0, w_undamped
    if !isfinite(condition) # The undamped solve did not succeed: the first damping that factorizes
        damping = NaN
        for λ in REC_RANK_DAMPINGS
            dampen!(ls, λ, d)
            rho, w = scaled_inverse_iteration(ls, linsolver, sqrtd)
            isfinite(rho) && (damping = λ; break)
        end
        isnan(damping) && return (:nofactorization, NaN, NaN, nothing, "")
    end
    J_auto, J, p = try
        rank_jacobian(model, hessorder; unfixed, numthreads, max_dense_bytes)
    catch err
        err isa ArgumentError || rethrow()
        return (:nojacobian, NaN, NaN, nothing, err.msg)
    end
    return (:ran, damping, minimum(rank_readings(J, p, sqrtd, w, ls, linsolver)), J_auto, "")
end

# Seconds taken by the second of two calls of `fun`, so that compilation is excluded
function secondcall_seconds(fun)
    fun()
    t0 = time_ns()
    fun()
    return (time_ns() - t0) * 1e-9
end

# Documented in docstrings.jl
function recommendoptimizer(model::Model; unfixed=nothing, numthreads=StaticInt(Threads.nthreads()), max_dense_bytes::Integer=MAX_DENSE_BYTES)
    # Stage 1, structure: the Hessian path exactly as `:auto` builds it
    f, fdf!, x0, H = optimvars(model; unfixed, numthreads, max_dense_bytes)
    conv = convertunfixed(unfixed, model)
    nvariables = conv isa Integer ? 1 : count(conv)
    n = unfixeddims(model, conv)
    ncosts = Int(countcosts(costnum, model.costs))
    allresiduals = all(v -> eltype(v) <: AbstractResidual, values(model.costs))
    m = sum((Int(dynamic(resnum(v))) for v in values(model.costs) if eltype(v) <: AbstractResidual); init=0)
    kind = hessiansystemkind(H)
    reduction, storage = reductionstorage(kind)
    retained, eliminated = elimdims(H)

    # Stage 2, the probe: the system the optimizer would build, linearized at the current values, then factorized
    linsolver, rule_l = rule_linsolver(allresiduals, kind)
    ls = isnothing(H) ? makehessiansystem(n, linsolver) : makehessiansystem(H, linsolver)
    linearize() = (DampedLinearSolve.zero!(ls); fdf!(DampedLinearSolve.getb(ls), DampedLinearSolve.getA(ls), x0))
    t_cost = secondcall_seconds(() -> f(x0))
    c = 0.0
    t_linearize = secondcall_seconds(() -> (c = linearize()))
    (isfinite(c) && isfinitesystem(ls)) || throw(ArgumentError("The model cannot be linearized at its current values: " *
        (isfinite(c) ? "its Hessian or gradient is not finite." : "its cost is $c.")))
    d = getdiag!(Vector{Float64}(undef, n), ls)
    diag_min, diag_max = extrema(d)
    diagspread = diag_min > 0 ? diag_max / diag_min : Inf
    status = Success
    t_solve = secondcall_seconds(() -> (status = solve!(ls, linsolver)))
    condition, w = (status == Success && diag_min > 0) ? scaled_inverse_iteration(ls, linsolver, sqrt.(d)) : (Inf, nothing)

    # The rank test of the Jacobian, exactly when rule F reads it
    ranktest, rank_damping, rank_t, J_auto, fallback = :notneeded, NaN, NaN, nothing, ""
    if needsranktest(allresiduals, status, condition)
        ranktest, rank_damping, rank_t, J_auto, fallback = jacobian_rank_test(model, ls, linsolver, d, condition, w,
            varorder(fdf!, model); unfixed, numthreads, max_dense_bytes)
    end

    features = RecommenderFeatures(nvariables, n, ncosts, m, allresiduals, kind, retained, eliminated,
        diag_min, diag_max, diagspread, status, condition, ranktest, rank_damping, rank_t, t_cost, t_linearize, t_solve)

    # The rules
    formulation, rule_f = rule_formulation(features)
    rule_rs = :RS_hessian
    if formulation === :jacobian
        # The Jacobian path as `:auto` builds it, from the prototype the rank test built
        reduction, storage = jacobianreductionstorage(J_auto)
        linsolver, rule_l = :qr, :L_jacobian
        rule_rs = :RS_jacobian
    end
    iterator, rule_i = rule_iterator(features)
    rules = (formulation = rule_f, reduction = rule_rs, storage = rule_rs, linsolver = rule_l, iterator = rule_i)
    return OptimizationRecommendation(formulation, reduction, storage, linsolver, iterator, unfixed, numthreads,
        Int(max_dense_bytes), features, rules, fallback)
end

# Display

function Base.show(io::IO, rec::OptimizationRecommendation)
    print(io, "OptimizationRecommendation(formulation = ", repr(rec.formulation), ", reduction = ", repr(rec.reduction),
        ", storage = ", repr(rec.storage), ", linsolver = ", repr(rec.linsolver), ", iterator = ", repr(rec.iterator), ")")
end

# Numbers for reading: three significant digits
fmtnum(x::Real) = isfinite(x) ? string(round(Float64(x); sigdigits=3)) : string(x)
function fmttime(t::Real)
    t < 1e-6 && return string(round(t * 1e9; sigdigits=3), " ns")
    t < 1e-3 && return string(round(t * 1e6; sigdigits=3), " μs")
    t < 1 && return string(round(t * 1e3; sigdigits=3), " ms")
    return string(round(t; sigdigits=3), " s")
end

# Prints `text` word-wrapped to `width` columns, each line indented by `indent` spaces
function printwrapped(io::IO, text::AbstractString, indent::Int, width::Int=80)
    col = 0
    for word in split(text)
        if col > 0 && col + 1 + textwidth(word) > width
            println(io)
            col = 0
        end
        if col == 0
            print(io, " "^indent, word)
            col = indent + textwidth(word)
        else
            print(io, " ", word)
            col += 1 + textwidth(word)
        end
    end
    println(io)
end

condition_text(f::RecommenderFeatures) = "the condition estimate of the Jacobi-scaled Hessian, $(fmtnum(f.condition))"

# Why rule F needed the rank test: the Hessian is ill-conditioned, or could not be factorized undamped
function illconditioned_text(f::RecommenderFeatures)
    isfinite(f.condition) && return "$(condition_text(f)), is at least $(fmtnum(REC_JACOBIAN_CONDITION))"
    damped = f.ranktest === :ran ? " (the test used a scaled damping of $(fmtnum(f.rank_damping)))" : ""
    return "the undamped Hessian could not be factorized ($(f.status))$damped"
end

# What the rank test read
rank_text(f::RecommenderFeatures) = "along the Hessian's weakest direction the column-scaled Jacobian has norm " *
    "$(fmtnum(f.rank_t)), $(f.rank_t < REC_RANK_TOLERANCE ? "below" : "at least") $(fmtnum(REC_RANK_TOLERANCE))"

function formulation_reason(rec::OptimizationRecommendation)
    f = rec.features
    rule = rec.rules.formulation
    rule === :F_nonresidual && return "Some costs are not residuals (AbstractResidual), so the model has no Jacobian: " *
        "the Hessian path is the only one."
    rule === :F_wellconditioned && return "Every cost is a residual, and $(condition_text(f)), is below " *
        "$(fmtnum(REC_JACOBIAN_CONDITION)): the normal equations are accurate enough."
    rule === :F_rankdeficient && return "Every cost is a residual, and $(illconditioned_text(f)), so the rank of the " *
        "Jacobian decides: $(rank_text(f)), so J is numerically rank-deficient (a gauge freedom, or a J whose own " *
        "condition number is beyond what Float64 resolves). Factorizing J would gain nothing, and damping handles it: the " *
        "Hessian path."
    rule === :F_illconditioned && return "Every cost is a residual, and $(illconditioned_text(f)), so the rank of the " *
        "Jacobian decides: $(rank_text(f)), so J has full numerical rank, and the condition number of the column-scaled " *
        "J is at least $(fmtnum(1 / f.rank_t)). Forming JᵀJ squares the conditioning of J, so the Jacobian is factorized " *
        "instead."
    rule === :F_singular && return (f.ranktest === :nodiagonal ?
        "The Hessian has a diagonal entry that is not positive (the smallest is $(fmtnum(f.diag_min)); the undamped " *
        "solve returned $(f.status)): a direction no residual moves, or an indefinite Hessian. " :
        "The Hessian could not be factorized ($(f.status)), undamped or with any scaled damping up to " *
        "$(fmtnum(last(REC_RANK_DAMPINGS))): it is rank-deficient or indefinite. ") *
        "The rank test of the Jacobian cannot be run on it, and damping handles it: the Hessian path."
    return "Every cost is a residual, and $(illconditioned_text(f)), so the rank of the Jacobian would decide, but the " *
        "Jacobian path cannot be built for this model ($(rec.fallback)), so the Hessian path."
end

function reduction_reason(rec::OptimizationRecommendation)
    f = rec.features
    path = rec.rules.reduction === :RS_jacobian ? "Jacobian" : "Hessian"
    rec.reduction === :eliminate || return "What reduction = :auto builds on the $path path: no variables are eliminated " *
        (f.nvariables == 1 ? "(a single unfixed variable never is)." : "(no eliminable variables, or a structure that does not allow it).")
    rec.rules.reduction === :RS_jacobian && return "What reduction = :auto builds on the Jacobian path: the eliminable " *
        "variables are eliminated by QR."
    return "What reduction = :auto builds on the Hessian path: $(f.eliminated) of the $(f.dimension) dimensions are " *
        "eliminated (the Schur complement), leaving a reduced system of $(f.retained)."
end

function storage_reason(rec::OptimizationRecommendation)
    f = rec.features
    if rec.rules.storage === :RS_jacobian
        what = rec.reduction === :eliminate ? "the reduced system" : "the $(f.nresiduals)×$(f.dimension) Jacobian"
        return "What storage = :auto builds on the Jacobian path: $what is stored $(rec.storage)."
    end
    f.hessiansystem === :static && return "What storage = :auto builds: one variable of static size, so a static " *
        "(stack-allocated) $(f.dimension)×$(f.dimension) system."
    what = rec.reduction === :eliminate ? "the $(f.retained)×$(f.retained) reduced system" :
        "the $(f.dimension)×$(f.dimension) Hessian"
    how = rec.reduction === :eliminate ? ", from the predicted fill of its Cholesky factor." :
        f.nvariables == 1 ? ", as a single variable's always is." : ", from its sparsity and size."
    return "What storage = :auto builds: $what is stored $(rec.storage)$how"
end

function linsolver_reason(rec::OptimizationRecommendation)
    rule = rec.rules.linsolver
    rule === :L_jacobian && return "QR of the Jacobian, on the Jacobian path."
    rule === :L_nonresidual && return "LDLᵀ: some costs are not residuals, so the Hessian can be indefinite."
    rule === :L_sparse && return "LDLᵀ, which linsolver = :auto uses for a sparse Hessian."
    rule === :L_schur && return "Cholesky of the reduced system (falling back to Bunch-Kaufman), which linsolver = :auto " *
        "uses for the Schur complement."
    return "Cholesky, which linsolver = :auto uses for a $(rule === :L_static ? "static" : "dense") Hessian."
end

function iterator_reason(rec::OptimizationRecommendation)
    f = rec.features
    rule = rec.rules.iterator
    rule === :I_dogleg && return "The undamped solve succeeded, $(condition_text(f)), is below " *
        "$(fmtnum(REC_DOGLEG_CONDITION)), the diagonal's spread (largest / smallest), $(fmtnum(f.diagspread)), is below " *
        "$(fmtnum(REC_DIAGSPREAD)), and a solve costs at least a linearization: dogleg, which does not solve again after " *
        "a rejected step."
    rule === :I_dtd && return "The diagonal's spread (largest / smallest), $(fmtnum(f.diagspread)), is at least " *
        "$(fmtnum(REC_DIAGSPREAD)): Levenberg-Marquardt with its damping scaled by the diagonal."
    t = dogleg_tests(f)
    failed = String[]
    t.success || push!(failed, "the undamped solve returned $(f.status)")
    t.condition || push!(failed, "$(condition_text(f)), is not below $(fmtnum(REC_DOGLEG_CONDITION))")
    t.solvecost || push!(failed, "a solve ($(fmttime(f.t_solve))) costs less than a linearization ($(fmttime(f.t_linearize)))")
    return "Levenberg-Marquardt: dogleg is not chosen because " * join(failed, ", and ") * "."
end

# The equivalent call to `optimize!(model, options; ...)`, as lines of code
function recommendation_call(rec::OptimizationRecommendation)
    lines = ["optimize!(model,", "    OptimizationOptions(iterator = $(repr(rec.iterator)),",
        "        linsolver = $(repr(rec.linsolver)));",
        "    formulation = $(repr(rec.formulation)), reduction = $(repr(rec.reduction)), storage = $(repr(rec.storage))"]
    extra = String[]
    if !isnothing(rec.unfixed)
        push!(extra, "unfixed = " * ((rec.unfixed isa Integer || rec.unfixed isa Type) ? string(rec.unfixed) : "unfixed"))
    end
    dynamic(rec.numthreads) == Threads.nthreads() || push!(extra, "numthreads = $(Int(dynamic(rec.numthreads)))")
    rec.max_dense_bytes == MAX_DENSE_BYTES || push!(extra, "max_dense_bytes = $(rec.max_dense_bytes)")
    if !isempty(extra)
        lines[end] *= ","
        push!(lines, "    " * join(extra, ", "))
    end
    lines[end] *= ")"
    return lines
end

function Base.show(io::IO, ::MIME"text/plain", rec::OptimizationRecommendation)
    f = rec.features
    printwrapped(io, "OptimizationRecommendation for $(f.nvariables) unfixed variable$(f.nvariables == 1 ? "" : "s") of " *
        "total dimension $(f.dimension), $(f.ncosts) cost$(f.ncosts == 1 ? "" : "s") and $(f.nresiduals) residual row" *
        "$(f.nresiduals == 1 ? "" : "s"), at the model's current values:", 0)
    for (name, reason) in ((:formulation, formulation_reason), (:reduction, reduction_reason), (:storage, storage_reason),
            (:linsolver, linsolver_reason), (:iterator, iterator_reason))
        println(io, "  ", rpad(name, 11), " = ", repr(getfield(rec, name)))
        printwrapped(io, reason(rec), 6)
    end
    printwrapped(io, "Measured once each, after a warm-up call: a cost evaluation $(fmttime(f.t_cost)), a linearization " *
        "$(fmttime(f.t_linearize)), an undamped solve $(fmttime(f.t_solve)). The thresholds are starting values, not " *
        "measured.", 0)
    println(io)
    println(io, "The same choices, by hand (with SecondOrderOptimization loaded):")
    (isnothing(rec.unfixed) || rec.unfixed isa Integer || rec.unfixed isa Type) ||
        println(io, "(`unfixed` stands for the value the recommendation was made with.)")
    lines = recommendation_call(rec)
    for (k, line) in enumerate(lines)
        k == length(lines) ? print(io, "  ", line) : println(io, "  ", line)
    end
end
