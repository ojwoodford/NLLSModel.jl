using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
import BlockLinearSystems
import DampedLinearSolve
import OptimizationVariableInterface

fixture("frontdoor", "optimization", "jacobianelimination", "nonsquaredcost", "adaptivecost", "jacobianproblems")

# Features built by hand, for the rules: a well-conditioned, well-scaled dense system whose solve costs more than its
# linearization, unless a keyword says otherwise. The rank test is consistent with the rest unless a keyword says
# otherwise: not needed when the Hessian is well conditioned; otherwise not possible with a non-positive diagonal, and
# run (undamped when the condition estimate is finite) with a full-rank reading
function recfeatures(; nvariables=2, dimension=4, ncosts=3, nresiduals=6, allresiduals=true, hessiansystem=:dense,
        retained=0, eliminated=0, diag_min=1.0, diag_max=10.0, diagspread=diag_max / diag_min,
        status=DampedLinearSolve.Success, condition=10.0,
        ranktest=(!NLLSModel.needsranktest(allresiduals, status, condition) ? :notneeded : diag_min > 0 ? :ran : :nodiagonal),
        rank_damping=(ranktest === :ran ? (isfinite(condition) ? 0.0 : 1e-14) : NaN),
        rank_t=(ranktest === :ran ? 1e-6 : NaN),
        work_hessian=(allresiduals ? 100.0 : NaN), work_jacobian=(allresiduals ? 10.0 : NaN),
        densejacobian=(all(NLLSModel.normalequations_tests(allresiduals, status, condition, hessiansystem, dimension,
            work_hessian, work_jacobian)) ? :available : :notneeded),
        t_cost=1e-6, t_linearize=1e-5, t_solve=2e-5)
    return NLLSModel.RecommenderFeatures(nvariables, dimension, ncosts, nresiduals, allresiduals, hessiansystem, retained,
        eliminated, diag_min, diag_max, diagspread, status, condition, ranktest, rank_damping, rank_t, work_hessian,
        work_jacobian, densejacobian, t_cost, t_linearize, t_solve)
end

# N2: `nvars` Vector{Float64} variables of `dim` dimensions each, and `rows` scalar linear residuals on each variable (a
# random, well-conditioned `X` per variable): with one variable, every cost touches every variable
function scalarmodel(nvars, dim, rows; seed=7)
    rng = MersenneTwister(seed)
    model = Model(Vector{Float64}, LinearResidualDynamic)
    for k in 1:nvars
        addvariable!(model, randn(rng, dim))
        X = randn(rng, rows, dim)
        for i in 1:rows
            addcost!(model, LinearResidualDynamic([randn(rng)], X[i:i, :], k))
        end
    end
    return model
end

# N2: an unfixed adaptive kernel with a few samples of its mean, beside a dense block on which the normal equations from J
# would cost less; well conditioned, but optimvars_nls refuses an unfixed adaptive kernel
function adaptivedensemodel(; samples=200, dim=20, rows=400)
    rng = MersenneTwister(1)
    model = Model(Union{ContaminatedGaussian{Float64}, Float64, Vector{Float64}}, Union{SimpleResidual, LinearResidualDynamic})
    addvariable!(model, ContaminatedGaussian(1.0, 5.0, 0.8))
    addvariable!(model, 0.1)
    addvariable!(model, randn(rng, dim))
    for _ in 1:samples
        addcost!(model, SimpleResidual(rand(rng) < 0.8 ? randn(rng) : 5 * randn(rng), 2))
    end
    X = randn(rng, rows, dim)
    for i in 1:rows
        addcost!(model, LinearResidualDynamic([randn(rng)], X[i:i, :], 3))
    end
    return model
end

# A linear least-squares model in one `Vector{Float64}` variable: r(x) = X x - y
function linearmodel(X, y=zeros(size(X, 1)), x0=ones(size(X, 2)))
    model = Model(Vector{Float64}, LinearResidualDynamic)
    addvariable!(model, x0)
    addcost!(model, LinearResidualDynamic(y, X, 1))
    return model
end

# X with XᵀX = S Hₛ S, S = Diagonal(s), Hₛ = (1 - c) I + c 11ᵀ: Hₛ is the Jacobi-scaled Hessian, with a unit diagonal and
# eigenvalues 1 - c (n - 1 times) and 1 + (n - 1) c, so its condition number is known by construction
scaledjacobian(n, c, s=ones(n)) = Matrix(cholesky(Symmetric(Matrix((1 - c) * I(n) + c * ones(n, n)))).U) * Diagonal(s)
scaledcondition(n, c) = (1 + (n - 1) * c) / (1 - c)

# A chain of `n` 2-vectors, each linked to the next and pinned to zero with weight `sqrt(ε)`: the scaled Hessian is a path
# graph's Laplacian plus ε, so it is sparse and, for small ε, finitely but badly conditioned
function weakchain(n, ε)
    model = Model(Vector{Float64}, Union{LinearResidualDynamic, Lnk})
    rng = MersenneTwister(5)
    for k in 1:n
        addvariable!(model, randn(rng, 2))
        addcost!(model, LinearResidualDynamic(zeros(2), sqrt(ε) * Matrix(1.0I, 2, 2), k))
    end
    for k in 1:n-1
        addcost!(model, Lnk(0.1 * randn(rng, 2), k, k + 1))
    end
    return model
end

# A residual and a general (non-residual) cost on one variable
function nonresidualmodel()
    model = Model(Vector{Float64}, Union{LinearResidualDynamic, LinearCostDynamic})
    addvariable!(model, ones(3))
    addcost!(model, LinearResidualDynamic(zeros(3), scaledjacobian(3, 0.5), 1))
    addcost!(model, LinearCostDynamic(ones(3), 1))
    return model
end

# An unfixed adaptive kernel (variable 1) fitted to 200 samples of a mixture through the mean (variable 2), and an
# ill-conditioned linear block on variable 3: the Hessian is positive definite and its scaled condition is above the
# formulation's threshold, but optimvars_nls refuses an unfixed adaptive kernel
function adaptivemodel()
    rng = MersenneTwister(1)
    model = Model(Union{ContaminatedGaussian{Float64}, Float64, Vector{Float64}}, Union{SimpleResidual, LinearResidualDynamic})
    addvariable!(model, ContaminatedGaussian(1.0, 5.0, 0.8))
    addvariable!(model, 0.1)
    addvariable!(model, ones(3))
    for _ in 1:200
        addcost!(model, SimpleResidual(rand(rng) < 0.8 ? randn(rng) : 5 * randn(rng), 2))
    end
    addcost!(model, LinearResidualDynamic(zeros(3), scaledjacobian(3, 1 - 1e-13), 3))
    return model
end

# The models, what the recommender reads from each, and what it should choose. `kind` is the Hessian path's system.
const REC_MODELS = (
    (name = "one static variable", make = () -> singleproblem(SVector{2, Float64}, 2, 1)[1],
        kind = :static, formulation = :hessian, linsolver = :cholesky),
    (name = "several variables, dense", make = () -> chainproblem(5, 2, 2)[1],
        kind = :dense, formulation = :hessian, linsolver = :cholesky),
    (name = "several variables, sparse", make = () -> chainproblem(60, 2, 2)[1],
        kind = :sparse, formulation = :hessian, linsolver = :ldl),
    (name = "eliminable, dense reduced system", make = () -> posechain(8, 12),
        kind = :schur_dense, formulation = :hessian, linsolver = :cholesky),
    (name = "eliminable, sparse reduced system", make = () -> posechain(200, 100),
        kind = :schur_sparse, formulation = :hessian, linsolver = :cholesky),
    (name = "a non-residual cost", make = nonresidualmodel,
        kind = :dense, formulation = :hessian, linsolver = :ldl),
    (name = "an unfixed adaptive kernel", make = adaptivemodel,
        kind = :dense, formulation = :hessian, linsolver = :cholesky),
    (name = "ill-conditioned, one variable", make = () -> linearmodel(scaledjacobian(3, 1 - 1e-13)),
        kind = :dense, formulation = :jacobian, linsolver = :qr),
    (name = "ill-conditioned, sparse", make = () -> weakchain(60, 1e-14),
        kind = :sparse, formulation = :jacobian, linsolver = :qr),
    # A gauge freedom (no prior): rounding lets the factorization succeed, with a finite condition estimate near 1/eps,
    # so the rank test of the Jacobian decides, and finds it rank-deficient (RC8)
    (name = "bundle adjustment, a gauge freedom", make = () -> (Random.seed!(3); optlandmark_problem(4, 20)),
        kind = :schur_dense, formulation = :hessian, linsolver = :cholesky),
)

# A linear least-squares model r(x) = X x with X = Q₁ Diagonal(σ) Q₂ᵀ, Q₁ and Q₂ random with orthonormal columns: its
# singular values are σ (before column scaling)
function svdmodel(m, σ; seed=1)
    rng = MersenneTwister(seed)
    n = length(σ)
    Q1 = Matrix(qr(randn(rng, m, n)).Q)[:, 1:n]
    Q2 = Matrix(qr(randn(rng, n, n)).Q)
    return linearmodel(Q1 * Diagonal(σ) * Q2', zeros(m), ones(n))
end

# A monomial fit of degree `deg` at `m` equally spaced points of [0, 1]: full rank, and badly conditioned
monomialmodel(m, deg) = linearmodel([x^k for x in range(0, 1; length=m), k in 0:deg])

# An unfixed adaptive kernel (variable 1) and the mean (variable 2) fitted to 200 samples of a mixture, at a point where
# the Hessian is indefinite with a positive diagonal (scaled eigenvalues about -4.2, 0.27, 1.2 and 6.7): no damping on
# the rank test's ladder factorizes it
function indefinitemodel()
    rng = MersenneTwister(1)
    model = Model(Union{ContaminatedGaussian{Float64}, Float64}, SimpleResidual)
    addvariable!(model, ContaminatedGaussian(1.0, 5.0, 0.8))
    addvariable!(model, 2.0)
    for _ in 1:200
        addcost!(model, SimpleResidual(rand(rng) < 0.8 ? randn(rng) : 5 * randn(rng), 2))
    end
    return model
end

# A curve fit with outliers under a kernel with a second-order correction that is not zero (Geman-McClure)
function gemanmcclurecurve()
    plain = curvefit(; outliers=5)
    k = GemanMcclure2oKernel(0.05)
    model = Model(Vector{Float64}, TlRob{TlExp, typeof(k)})
    addvariable!(model, plain.variables[1])
    foreach(c -> addcost!(model, TlRob(c, k)), Iterators.flatten(values(plain.costs)))
    return model
end

# A linear residual r(x) = X x - y on one variable of static size N, so that the Hessian path builds a static system
struct StaticLinRes{N} <: NLLSModel.AbstractResidual
    X::Matrix{Float64}
    y::Vector{Float64}
end
NLLSModel.ndeps(::StaticLinRes) = static(1)
NLLSModel.nres(r::StaticLinRes) = length(r.y)
NLLSModel.varindices(::StaticLinRes) = 1
NLLSModel.getvars(::StaticLinRes{N}, vars::AbstractVector) where N = (vars[1]::SVector{N, Float64},)
NLLSModel.computeresidual(r::StaticLinRes, x) = r.X * x - r.y
Base.eltype(::StaticLinRes) = Float64
function staticlinearmodel(X)
    N = size(X, 2)
    model = Model(SVector{N, Float64}, StaticLinRes{N})
    addvariable!(model, SVector{N, Float64}(ones(N)))
    addcost!(model, StaticLinRes{N}(X, zeros(size(X, 1))))
    return model
end

# The models of the rank test (RC8), what it should find, and the formulation rule that should fire. `damping` is the
# scaled damping of the factorization the test used.
const REC_RANK_MODELS = (
    (name = "full rank, cond(J) 3e7, factorizes undamped", make = () -> svdmodel(40, exp10.(range(0, -7.5; length=10))),
        rule = :F_illconditioned, ranktest = :ran, damping = 0.0),
    (name = "full rank, a degree-14 monomial fit, does not factorize undamped", make = () -> monomialmodel(400, 14),
        rule = :F_illconditioned, ranktest = :ran, damping = 1e-14),
    (name = "a 7-dimensional null space, remainder to 1e-4", make = () -> svdmodel(60, [exp10.(range(0, -4; length=13)); zeros(7)]),
        rule = :F_rankdeficient, ranktest = :ran, damping = 1e-14),
    (name = "a 7-dimensional null space, remainder to 1e-6", make = () -> svdmodel(60, [exp10.(range(0, -6; length=13)); zeros(7)]),
        rule = :F_rankdeficient, ranktest = :ran, damping = 1e-14),
    (name = "full rank, cond(J) 3e7, static, factorizes undamped",
        make = () -> staticlinearmodel(only(values(svdmodel(40, exp10.(range(0, -7.5; length=10))).costs))[1].X),
        rule = :F_illconditioned, ranktest = :ran, damping = 0.0),
    (name = "full rank, a degree-14 monomial fit, static, does not factorize undamped",
        make = () -> staticlinearmodel([x^k for x in range(0, 1; length=400), k in 0:14]),
        rule = :F_illconditioned, ranktest = :ran, damping = 1e-14),
    (name = "a zero column", make = () -> linearmodel([1.0 0.0; 2.0 0.0]),
        rule = :F_singular, ranktest = :nodiagonal, damping = NaN),
    (name = "indefinite, an unfixed adaptive kernel", make = indefinitemodel,
        rule = :F_singular, ranktest = :nofactorization, damping = NaN),
)

# The pieces of the rank test for a model, as `recommendoptimizer` builds them: the Hessian system linearized at the
# current values (undamped, not yet solved), its diagonal, the Jacobian evaluated there and the permutation from the
# Hessian's order to the Jacobian's columns
function rankpieces(model; unfixed=nothing, kwargs...)
    _, fdf!, x0, H = optimvars(model; unfixed, numthreads=static(1), kwargs...)
    n = NLLSModel.unfixeddims(model, NLLSModel.convertunfixed(unfixed, model))
    linsolver = NLLSModel.rule_linsolver(true, NLLSModel.hessiansystemkind(H))[1]
    ls = isnothing(H) ? DampedLinearSolve.makehessiansystem(n, linsolver) : DampedLinearSolve.makehessiansystem(H, linsolver)
    linearizehessian!(ls, fdf!, x0)
    d = DampedLinearSolve.getdiag!(zeros(n), ls)
    _, J, p = NLLSModel.rank_jacobian(model, NLLSModel.varorder(fdf!, model); unfixed, numthreads=static(1),
        max_dense_bytes=NLLSModel.MAX_DENSE_BYTES)
    return (; ls, linsolver, d, J, p, fdf!)
end

# ‖J D^-½ w‖² and wᵀ D^-½ H D^-½ w for a random unit `w` in the Hessian's order, with the permutation `p`
function permutedproducts(pieces, p=pieces.p)
    w = normalize(randn(MersenneTwister(4), length(pieces.d)))
    v = zeros(size(pieces.J, 2))
    v[p] .= w ./ sqrt.(pieces.d)
    return norm(pieces.J * v)^2, DampedLinearSolve.quadform(DampedLinearSolve.getA(pieces.ls), w ./ sqrt.(pieces.d))
end

# The rank test's readings for a model, without and with the refinement step: the undamped direction when the Hessian
# factorizes undamped, otherwise the first damping that does
function rankreadings(model)
    pc = rankpieces(model)
    sqrtd = sqrt.(pc.d)
    rho, w = NLLSModel.scaled_inverse_iteration(pc.ls, pc.linsolver, sqrtd)
    for λ in NLLSModel.REC_RANK_DAMPINGS
        isfinite(rho) && break
        DampedLinearSolve.dampen!(pc.ls, λ, pc.d)
        rho, w = NLLSModel.scaled_inverse_iteration(pc.ls, pc.linsolver, sqrtd)
    end
    return NLLSModel.rank_readings(pc.J, pc.p, sqrtd, w, pc.ls, pc.linsolver)
end

# The singular values of the column-scaled J, largest first
scaledsvdvals(J) = (Jd = Matrix(J); svdvals(Jd ./ sqrt.(sum(abs2, Jd; dims=1))))
scaledsigmamin(J) = last(scaledsvdvals(J))

# The features that do not depend on timing
untimed(f::NLLSModel.RecommenderFeatures) = Tuple(getfield(f, k) for k in fieldnames(NLLSModel.RecommenderFeatures) if !startswith(string(k), "t_"))

# Whether the iterator was decided by the timings: every other test of dogleg's rule passes
timingdecided(f::NLLSModel.RecommenderFeatures) = (t = NLLSModel.dogleg_tests(f); t.success && t.condition && t.diagspread)

# Linearizes a Hessian system as SecondOrderOptimization does, at the model's current values
function linearizehessian!(ls, fdf!, x0)
    DampedLinearSolve.zero!(ls)
    return fdf!(DampedLinearSolve.getb(ls), DampedLinearSolve.getA(ls), x0)
end

# The undamped step of a Hessian system built with `linsolver`, from the prototype of `optimvars`
function hessianstep(model, linsolver)
    _, fdf!, x0, H = optimvars(model; numthreads=static(1))
    n = NLLSModel.unfixeddims(model, NLLSModel.convertunfixed(nothing, model))
    ls = isnothing(H) ? DampedLinearSolve.makehessiansystem(n, linsolver) : DampedLinearSolve.makehessiansystem(H, linsolver)
    linearizehessian!(ls, fdf!, x0)
    status = DampedLinearSolve.solve!(ls, linsolver)
    return status, copy(DampedLinearSolve.getx(ls))
end

# The undamped step of the Jacobian system built with `linsolver`, from the prototype of `optimvars_nls`
function jacobianstep(model, linsolver; kwargs...)
    _, _, resjac!, x0, J = optimvars_nls(model; numthreads=static(1), kwargs...)
    n = NLLSModel.unfixeddims(model, NLLSModel.convertunfixed(nothing, model))
    ls = DampedLinearSolve.makejacobiansystem(J, n, linsolver)
    DampedLinearSolve.zero!(ls)
    A = DampedLinearSolve.getA(ls)
    resjac!(DampedLinearSolve.getr(A), DampedLinearSolve.getJ(A), x0)
    DampedLinearSolve.gradient!(DampedLinearSolve.getb(ls), A)
    DampedLinearSolve.markstale!(A)
    status = DampedLinearSolve.solve!(ls, linsolver)
    return status, copy(DampedLinearSolve.getx(ls))
end

# A recommendation built by hand, for `show`: the rules' symbols default to those its choices and features imply
function recby(; features=recfeatures(), formulation=:hessian, reduction=:none, storage=:dense, linsolver=:cholesky,
        iterator=:levenbergmarquardt, unfixed=nothing, numthreads=static(Threads.nthreads()),
        max_dense_bytes=NLLSModel.MAX_DENSE_BYTES, fallback="",
        rule_f=NLLSModel.rule_formulation(features)[2],
        rule_rs=(rule_f === :F_normalequations ? :RS_jacobian_dense : formulation === :jacobian ? :RS_jacobian : :RS_hessian),
        rule_l=(rule_f === :F_normalequations ? :L_jacobian_normal : formulation === :jacobian ? :L_jacobian :
            NLLSModel.rule_linsolver(features.allresiduals, features.hessiansystem)[2]),
        rule_i=NLLSModel.rule_iterator(features)[2])
    rules = (formulation = rule_f, reduction = rule_rs, storage = rule_rs, linsolver = rule_l, iterator = rule_i)
    return NLLSModel.OptimizationRecommendation(formulation, reduction, storage, linsolver, iterator, unfixed, numthreads,
        Int(max_dense_bytes), features, rules, fallback)
end

longform(rec) = sprint(show, MIME"text/plain"(), rec)
flat(text) = join(split(text), " ") # Whitespace collapsed, so wrapped text can be searched

# The call `show` prints, parsed: its positional arguments, its keywords and the keywords of its `OptimizationOptions`
function printedcall(rec)
    text = longform(rec)
    ex = Meta.parse(text[first(findfirst("optimize!(model,", text)):end])
    (ex isa Expr && ex.head === :call && ex.args[1] === :optimize! && ex.args[2].head === :parameters) || return nothing
    keywords = Dict(a.args[1] => a.args[2] for a in ex.args[2].args)
    positional = ex.args[3:end]
    options = positional[2]
    (options isa Expr && options.head === :call && options.args[1] === :OptimizationOptions) || return nothing
    return positional, keywords, Dict(a.args[1] => a.args[2] for a in options.args[2:end])
end

# Runs the call `show` prints on `model`, with SecondOrderOptimization (loaded by the frontdoor fixture), and returns its
# result
function runprintedcall(rec, model)
    text = longform(rec)
    ex = Meta.parse(text[first(findfirst("optimize!(model,", text)):end])
    return Core.eval(@__MODULE__, :(let model = $model; $ex; end))
end
