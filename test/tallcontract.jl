using NLLSModel, Test, Static, StaticArrays, Random, LinearAlgebra
import OptimizationVariableInterface
import OptimizationVariableInterface: nvars, update, update!, copyvars

# T4: the contract the optimizer relies on. Needs tallproblems.jl.

@testset "tallcontract.jl" begin
    @testset "contract (T4): resjac! returns the true cost" begin
        for name in ("Huber", "Huber, mixed variables")
            for kernel in (true, :second)
                model = name == "Huber" ? curvefit(huber=kernel, outliers=5) : posemodel(huber=kernel)
                f, _, resjac!, x0, J = tlnls(model; numthreads=static(1), matrix_type=:dense)
                m, n = size(J)
                r = zeros(m)
                c = resjac!(r, J, x0)
                @test c ≈ NLLSModel.cost(model) rtol=1e-12 # The model holds x0
                @test c ≈ f(x0)
                @test c ≈ NLLSModel.cost(model.variables, model.costs)
                @test abs(c - 0.5 * sum(abs2, r)) > 1e-8 * c # Not the cost of the (reweighted) residual
                # At another point too
                x1 = update!(deepcopy(x0), x0, 0.05 .* ones(n))
                c1 = resjac!(r, J, x1)
                @test c1 ≈ f(x1)
                @test abs(c1 - 0.5 * sum(abs2, r)) > 1e-8 * c1
            end
        end
        # With no robust kernel, the cost is half the squared norm of the residual
        model = posemodel()
        f, _, resjac!, x0, J = tlnls(model; numthreads=static(1), matrix_type=:dense)
        r = zeros(size(J, 1))
        c = resjac!(r, J, x0)
        @test c ≈ 0.5 * sum(abs2, r)
        @test c ≈ f(x0)
    end

    # Without a robust kernel, the columns of J are the derivatives of r along the optimizer's update, in order: for every
    # variable, `nvars` columns, in the order of x0, and x0 follows `unfixed`.
    @testset "contract (T4): the columns of J are the optimizer's update order, $name" for (name, make, unfixed) in (
            ("curve fit", () -> curvefit(), nothing),
            ("rotation", () -> rotproblem(), nothing),
            ("all four variables", () -> posemodel(), nothing),
            ("a list", () -> posemodel(), [2, 3, 4]),
            ("a list out of order of size", () -> posemodel(), [4, 2]),
            ("a mask", () -> posemodel(), Bool[1, 0, 1, 1]),
            ("a type", () -> posemodel(), Vector{Float64}),
            ("a rotation alone", () -> posemodel(), 2),
            ("an SVector alone", () -> posemodel(), 3),
            ("a rotation and a Vector of 3", () -> posemodel(), [1, 2]))
        model = make()
        f, res!, resjac!, x0, J = tlnls(model; unfixed, numthreads=static(1), matrix_type=:dense)
        m, n = size(J)
        nx = x0 isa AbstractVector && !(x0 isa Vector{Float64} && length(x0) == n && !(eltype(x0) <: AbstractVector)) ? sum(v -> Int(dynamic(nvars(v))), x0) : Int(dynamic(nvars(x0)))
        @test n == nx
        resjac!(zeros(m), J, x0)
        J0 = copy(J)
        ε = 1e-6
        rp, rm = zeros(m), zeros(m)
        fderr = zeros(n, n) # fderr[j, k]: how far column k of J is from the finite difference along step j
        for j in 1:n
            e = zeros(n); e[j] = ε
            res!(rp, update!(deepcopy(x0), x0, e)) # The optimizer's own update!, as the iterators apply it
            res!(rm, update!(deepcopy(x0), x0, -e))
            fd = (rp .- rm) ./ (2ε)
            for k in 1:n
                fderr[j, k] = norm(fd .- J0[:, k], Inf) / max(1.0, norm(J0[:, k], Inf))
            end
        end
        @test all(j -> fderr[j, j] < 1e-5, 1:n)
        # and not merely close to a permutation of the columns: no other column matches unless it is the same column
        for j in 1:n, k in 1:n
            j == k || norm(J0[:, j] .- J0[:, k], Inf) < 1e-3 || @test fderr[j, k] > 1e-5
        end
    end

    # With a robust kernel, the rows of each cost are scaled by a scalar (IRLS weights are held fixed in J, so r is not
    # differentiated by finite differences). The columns are those of the same model with a kernel too wide to act.
    @testset "contract (T4): robust Jacobians have the columns of the plain model, $name" for (name, plain, robustmodel, unfixed) in (
            ("curve fit", h -> curvefit(huber=h, seed=4, outliers=3, width=1e8), h -> curvefit(huber=h, seed=4, outliers=3), nothing),
            ("all four variables", h -> posemodel(huber=h, width=1e8), h -> posemodel(huber=h), nothing),
            ("a list", h -> posemodel(huber=h, width=1e8), h -> posemodel(huber=h), [4, 1, 2]))
        for mode in (:irls, :corrector)
            _, _, rjp, x0p, Jp = tlnls(plain(mode === :irls ? true : :second); unfixed, numthreads=static(1), matrix_type=:dense)
            _, _, rjr, x0r, Jr = tlnls(robustmodel(mode === :irls ? true : :second); unfixed, numthreads=static(1), matrix_type=:dense)
            rp, rr = zeros(size(Jp, 1)), zeros(size(Jr, 1))
            rjp(rp, Jp, x0p)
            rjr(rr, Jr, x0r)
            if mode === :irls
                # Row i of the robust Jacobian is the plain one, scaled as the residual is
                rows = findall(abs.(rp) .> 1e-9)
                @test length(rows) > 5
                @test all(i -> Jr[i, :] ≈ (rr[i] / rp[i]) .* Jp[i, :], rows)
            else
                # The corrector also projects, so compare what it must keep: the gradient direction Jᵀr per cost
                @test size(Jr) == size(Jp)
                @test all(isfinite, Jr)
            end
        end
    end

    # Z3: with every variable unfixed, x0 *is* model.variables. Evaluating at another point must not overwrite it.
    staticmodel() = (rng = MersenneTwister(5); m = Model(SVector{2, Float64}, Tgt{SVector{2, Float64}});
        foreach(k -> addvariable!(m, randn(rng, SVector{2, Float64})), 1:3);
        foreach(k -> addcost!(m, Tgt(randn(rng, SVector{2, Float64}), k)), 1:3); m)
    @testset "Z3: evaluating at another point leaves x0 alone, $name" for (name, make) in (
            ("static blocks", staticmodel),
            ("Vectors", () -> first(chainproblem(4, 3, 6))),
            ("mixed, with a rotation", () -> posemodel()))
        model = make()
        _, res!, resjac!, x0, J = optimvars_nls(model; numthreads=static(1), matrix_type=:dense)
        @test x0 === model.variables
        snapshot = deepcopy(x0)
        m, n = size(J)
        y = update!(deepcopy(x0), x0, 0.1 .* ones(n))
        @test y != x0
        r = zeros(m)
        resjac!(r, J, y)
        @test x0 == snapshot
        res!(r, y)
        @test x0 == snapshot
        # f has never written
        optimvars_nls(model; numthreads=static(1))[1](y)
        @test x0 == snapshot
        # and the evaluation was at y: the residual matches a model built at y
        other = deepcopy(model)
        other.variables .= y
        _, _, rj2, x2, J2 = optimvars_nls(other; numthreads=static(1), matrix_type=:dense)
        r2 = zeros(m)
        rj2(r2, J2, x2)
        resjac!(r, J, y)
        @test r == r2 && J == J2
    end

    @testset "Z3: evaluating at x0 itself works and copies nothing" begin
        for model in (staticmodel(), first(chainproblem(4, 3, 6)))
            f, res!, resjac!, x0, J = optimvars_nls(model; numthreads=static(1), matrix_type=:dense)
            r = zeros(size(J, 1))
            c = resjac!(r, J, x0)
            @test c ≈ f(x0) && c ≈ NLLSModel.cost(model)
            resjac!(r, J, x0); res!(r, x0)
            # The evaluators do allocate (a few kB per call, independent of the variables); evaluating at x0 costs no more than
            # anywhere else, i.e. there is no copy of the variables
            y = update!(deepcopy(x0), x0, 0.1 .* ones(size(J, 2)))
            resjac!(r, J, y)
            @test (@allocated resjac!(r, J, x0)) <= (@allocated resjac!(r, J, y))
            @test (@allocated res!(r, x0)) <= (@allocated res!(r, y))
        end
    end

    @testset "Z3: a subset or a single variable is still merged into the model, $name" for (name, unfixed) in (("a list", [1, 3]), ("a mask", Bool[1, 0, 1, 1]), ("an index", 2), ("a type", Vector{Float64}))
        model = posemodel()
        _, res!, resjac!, x0, J = optimvars_nls(model; unfixed, numthreads=static(1), matrix_type=:dense)
        @test x0 !== model.variables
        fixed = [k for k in 1:4 if !(NLLSModel.convertunfixed(unfixed, model) isa Integer ? k == unfixed : NLLSModel.convertunfixed(unfixed, model)[k])]
        before = deepcopy(model.variables)
        x0c = deepcopy(x0)
        y = update!(deepcopy(x0), x0, 0.1 .* ones(size(J, 2)))
        r = zeros(size(J, 1))
        resjac!(r, J, y)
        @test x0 == x0c # The optimizer's vector is untouched
        for k in fixed
            @test model.variables[k] == before[k]
        end
        # the model holds y in the unfixed slots, so its cost is the cost at y
        ctot = NLLSModel.cost(model)
        @test ctot ≈ optimvars_nls(model; unfixed, numthreads=static(1))[1](y) rtol=1e-12
    end

    # Z1: the optimizer hands resjac! a J that started as zeros and does not zero it again (prezeroed = true: the structural zeros
    # are in place). Every call must therefore write every entry that can be non-zero.
    cutoffmodel() = (rng = MersenneTwister(8); m = Model(Vector{Float64}, TlRob{TlExp, TlCutoff});
        addvariable!(m, [2.0, 1.3, 0.5]);
        foreach(t -> addcost!(m, TlRob(TlExp(t, 2.0 * exp(-1.3 * t) + 0.5 + 0.3 * randn(rng), 1), TlCutoff(0.01))), range(0, 3; length=20)); m)
    nanmodel() = (m = Model(Vector{Float64}, Union{TlExp, TlNan}); addvariable!(m, [2.0, 1.3, 0.5]); addvariable!(m, [1.0, 0.0]);
        foreach(t -> addcost!(m, TlExp(t, 2.0 * exp(-1.3 * t) + 0.5, 1)), range(0, 3; length=8)); addcost!(m, TlNan(2)); m)

    # Points: x0 moved by steps of the given scales
    function repeatedpoints(x0, n, steps)
        xs = Any[x0]
        for s in steps
            push!(xs, update!(deepcopy(x0), x0, s .* ones(n)))
        end
        return xs
    end

    @testset "Z1: prezeroed = true into one J matches a fresh zeroed J at every point, $name" for (name, make, unfixed, steps) in (
            ("costs on different variable subsets", () -> posemodel(), nothing, (0.2, -0.3, 0.0, 0.5, -0.1)),
            ("a subset unfixed", () -> posemodel(), [1, 3], (0.2, -0.3, 0.0, 0.5, -0.1)),
            ("a single variable", () -> posemodel(), 2, (0.2, -0.3, 0.0, 0.5, -0.1)),
            ("robust, second order", () -> posemodel(huber=:second), nothing, (0.2, -0.3, 0.0, 0.5)),
            ("a weight that is zero at one point and not at the next", () -> cutoffmodel(), nothing, (0.0, 0.4, 0.0, -0.3, 0.9, 0.0)),
            ("non-finite, then finite, then non-finite", () -> nanmodel(), nothing, (0.0, 9.0, 0.0, 9.0, 9.0, 0.0)))
        model = make()
        _, _, resjac!, x0, J = optimvars_nls(model; unfixed, numthreads=static(1), matrix_type=:dense)
        m, n = size(J)
        @test all(iszero, J) # The prototype is zeros: the structural zeros are in place
        if name == "non-finite, then finite, then non-finite"
            # step 9 on variable 1's first entry makes TlNan non-finite; the others are steps of a different size
            xs = [x0, update!(deepcopy(x0), x0, [zeros(3); 9.0; 0.0]), x0, update!(deepcopy(x0), x0, [zeros(3); 9.0; 0.0]),
                  update!(deepcopy(x0), x0, [zeros(3); 9.0; 1.0]), update!(deepcopy(x0), x0, [0.1, 0, 0, 0, 0])]
        else
            xs = repeatedpoints(x0, n, steps)
        end
        Jreuse, rreuse = zeros(m, n), zeros(m)
        nonfinite_seen, zero_weight_seen = false, false
        for x in xs
            fill!(rreuse, 7.0) # r is not zeroed either: every entry is written
            c1 = resjac!(rreuse, Jreuse, x; prezeroed=true)
            Jfresh, rfresh = zeros(m, n), zeros(m)
            c2 = resjac!(rfresh, Jfresh, x)
            @test isequal(Jreuse, Jfresh)
            @test isequal(rreuse, rfresh)
            @test isequal(c1, c2)
            nonfinite_seen |= any(!isfinite, Jfresh)
            zero_weight_seen |= any(iszero, rfresh) && any(!iszero, rfresh)
        end
        name == "non-finite, then finite, then non-finite" && @test nonfinite_seen
        startswith(name, "a weight") && @test zero_weight_seen
        # J was only ever written by the evaluator, and an evaluation at a finite point leaves nothing from a non-finite one
        name == "non-finite, then finite, then non-finite" && @test all(isfinite, Jreuse) == all(isfinite, resjac!(zeros(m), zeros(m, n), xs[end]))
    end

    @testset "Z1: a weight that is zero at one point and non-zero at the next, and the reverse" begin
        model = cutoffmodel()
        _, _, resjac!, x0, J = optimvars_nls(model; numthreads=static(1), matrix_type=:dense)
        m, n = size(J)
        weights(x) = (r = zeros(m); Jt = zeros(m, n); resjac!(r, Jt, x); count(iszero, r))
        # Two points with different sets of zero weights, both mixed
        cands = [update!(deepcopy(x0), x0, d .* ones(3)) for d in -0.6:0.05:0.6]
        counts = weights.(cands)
        @test any(c -> 0 < c < m, counts) && length(unique(counts)) > 2
        far = cands[findfirst(c -> 0 < c < m, counts)]
        x0 = cands[findlast(c -> 0 < c < m && c != weights(far), counts)]
        @test weights(x0) != weights(far)
        Jr, r = zeros(m, n), zeros(m)
        for x in (far, x0, far, x0)
            resjac!(r, Jr, x; prezeroed=true)
            Jf, rf = zeros(m, n), zeros(m)
            resjac!(rf, Jf, x)
            @test Jr == Jf && r == rf
        end
    end

    @testset "Z1: optimvars_nls's dense prototype is all zeros, $name" for (name, make, unfixed) in (
            ("curve fit", () -> curvefit(), nothing), ("mixed", () -> posemodel(), nothing), ("a subset", () -> posemodel(), [2, 4]),
            ("static blocks", staticmodel, nothing))
        _, _, _, _, J = optimvars_nls(make(); unfixed, matrix_type=:dense)
        @test J isa Matrix{Float64}
        @test all(iszero, J)
        _, _, _, _, Ja = optimvars_nls(make(); unfixed) # :auto picks dense for these small models
        Ja isa Matrix && @test all(iszero, Ja)
    end
end
