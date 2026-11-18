# The optimizer recommender: `recommendoptimizer(model)` chooses a formulation, reduction, storage, linear solver and iterator
# by hand-written rules, from the model's structure and from one linearization and factorization of its Hessian system at
# the model's current values (improvements.md item 25). The rules are pure functions of the features, below, so that they
# can be tested without a model and without timing noise.

# The thresholds of the rules. Every one is a starting value, not measured (rules.md 26); they are listed for measurement
# in improvements.md item 25.
const REC_TALL_CONDITION = 1e12       # F: the Jacobian path at or above this condition estimate. Starting value, not measured.
const REC_DOGLEG_CONDITION = 1e8      # I: dogleg only below this condition estimate. Starting value, not measured.
const REC_DIAGSPREAD = 1e6            # I: dogleg only below, and D'D-scaled damping at or above, this spread of the
                                      # Hessian's diagonal. Starting value, not measured.
const REC_DOGLEG_SOLVE_RATIO = 1.0    # I: dogleg only when a solve costs at least this many linearizations. Starting
                                      # value, not measured.
const REC_INVERSE_ITERATIONS = 2      # Steps of inverse iteration in the condition estimate. Starting value, not measured.

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
elimdims(H::SchurBlockHessian) = (H.structure.nr, sum(Int, H.structure.e_dims; init=0))
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

# Rule F, the formulation: the Jacobian path when every cost is a residual and the scaled Hessian is finitely but badly
# conditioned. A system that could not be factorized (`condition = Inf`, rank-deficient or indefinite) stays on the Hessian
# path: a gauge freedom is no better conditioned as a Jacobian, and damping handles it.
function rule_formulation(f::RecommenderFeatures)
    f.allresiduals || return (:normal, :F_nonresidual)
    isfinite(f.condition) || return (:normal, :F_singular)
    f.condition >= REC_TALL_CONDITION && return (:tall, :F_illconditioned)
    return (:normal, :F_wellconditioned)
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

# A lower bound on the condition number of the Jacobi-scaled matrix D^-½ A D^-½ (D the diagonal of A), by inverse iteration
# on it, which needs only solves of the system: b = D^½ w, solve, y = D^½ x, ρ = ‖y‖ / ‖w‖, w = y / ‖y‖. The scaled matrix
# has a unit diagonal, so its largest eigenvalue is at least 1, and ρ, at most the largest eigenvalue of its inverse, is
# at most its condition number. Overwrites the system's b and x.
function scaled_condition_estimate(ls, linsolver::Symbol, d::Vector{Float64})
    sqrtd = sqrt.(d)
    w = inverse_iteration_start(length(d))
    rho = Inf
    for _ in 1:REC_INVERSE_ITERATIONS
        DampedLinearSolve.getb(ls) .= sqrtd .* w
        solve!(ls, linsolver) == Success || return Inf
        y = sqrtd .* DampedLinearSolve.getx(ls)
        rho = norm(y) # ‖w‖ = 1; positive, since b is not zero
        w = y ./ rho
    end
    return rho
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
    condition = (status == Success && diag_min > 0) ? scaled_condition_estimate(ls, linsolver, d) : Inf

    features = RecommenderFeatures(nvariables, n, ncosts, m, allresiduals, kind, retained, eliminated,
        diag_min, diag_max, diagspread, status, condition, t_cost, t_linearize, t_solve)

    # The rules
    formulation, rule_f = rule_formulation(features)
    rule_rs = :RS_hessian
    fallback = ""
    if formulation === :tall
        try
            J = optimvars_nls(model; unfixed, numthreads, max_dense_bytes)[5]
            reduction, storage = jacobianreductionstorage(J)
            linsolver, rule_l = :qr, :L_tall
            rule_rs = :RS_jacobian
        catch err
            err isa ArgumentError || rethrow()
            formulation, rule_f, fallback = :normal, :F_tall_unavailable, err.msg
        end
    end
    iterator, rule_i = rule_iterator(features)
    rules = (formulation = rule_f, reduction = rule_rs, storage = rule_rs, linsolver = rule_l, iterator = rule_i)
    return OptimizationRecommendation(formulation, reduction, storage, linsolver, iterator, unfixed, numthreads,
        Int(max_dense_bytes), features, rules, fallback)
end
