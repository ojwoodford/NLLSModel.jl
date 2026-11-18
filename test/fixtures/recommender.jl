using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
import BlockLinearSystems
import DampedLinearSolve
import OptimizationVariableInterface

fixture("frontdoor", "optimization", "tallelimination", "nonsquaredcost", "adaptivecost")

# Features built by hand, for the rules: a well-conditioned, well-scaled dense system whose solve costs more than its
# linearization, unless a keyword says otherwise
function recfeatures(; nvariables=2, dimension=4, ncosts=3, nresiduals=6, allresiduals=true, hessiansystem=:dense,
        retained=0, eliminated=0, diag_min=1.0, diag_max=10.0, diagspread=diag_max / diag_min,
        status=DampedLinearSolve.Success, condition=10.0, t_cost=1e-6, t_linearize=1e-5, t_solve=2e-5)
    return NLLSModel.RecommenderFeatures(nvariables, dimension, ncosts, nresiduals, allresiduals, hessiansystem, retained,
        eliminated, diag_min, diag_max, diagspread, status, condition, t_cost, t_linearize, t_solve)
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
        kind = :static, formulation = :normal, linsolver = :cholesky),
    (name = "several variables, dense", make = () -> chainproblem(5, 2, 2)[1],
        kind = :dense, formulation = :normal, linsolver = :cholesky),
    (name = "several variables, sparse", make = () -> chainproblem(60, 2, 2)[1],
        kind = :sparse, formulation = :normal, linsolver = :ldl),
    (name = "eliminable, dense reduced system", make = () -> posechain(8, 12),
        kind = :schur_dense, formulation = :normal, linsolver = :cholesky),
    (name = "eliminable, sparse reduced system", make = () -> posechain(200, 100),
        kind = :schur_sparse, formulation = :normal, linsolver = :cholesky),
    (name = "a non-residual cost", make = nonresidualmodel,
        kind = :dense, formulation = :normal, linsolver = :ldl),
    (name = "an unfixed adaptive kernel", make = adaptivemodel,
        kind = :dense, formulation = :normal, linsolver = :cholesky),
    (name = "ill-conditioned, one variable", make = () -> linearmodel(scaledjacobian(3, 1 - 1e-13)),
        kind = :dense, formulation = :tall, linsolver = :qr),
    (name = "ill-conditioned, sparse", make = () -> weakchain(60, 1e-14),
        kind = :sparse, formulation = :tall, linsolver = :qr),
    # A gauge freedom (no prior): rounding lets the factorization succeed, with a finite condition estimate near 1/eps,
    # so the formulation rule sends it to the Jacobian path (measured; see the Status of RC in NLLSModel.jl-fixes.md)
    (name = "bundle adjustment, a gauge freedom", make = () -> (Random.seed!(3); optlandmark_problem(4, 20)),
        kind = :schur_dense, formulation = :tall, linsolver = :qr),
)

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

# The undamped step of the tall system built with `linsolver`, from the prototype of `optimvars_nls`
function tallstep(model, linsolver; kwargs...)
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
function recby(; features=recfeatures(), formulation=:normal, reduction=:none, storage=:dense, linsolver=:cholesky,
        iterator=:levenbergmarquardt, unfixed=nothing, numthreads=static(Threads.nthreads()),
        max_dense_bytes=NLLSModel.MAX_DENSE_BYTES, fallback="",
        rule_f=NLLSModel.rule_formulation(features)[2], rule_rs=(formulation === :tall ? :RS_jacobian : :RS_hessian),
        rule_l=(formulation === :tall ? :L_tall : NLLSModel.rule_linsolver(features.allresiduals, features.hessiansystem)[2]),
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
