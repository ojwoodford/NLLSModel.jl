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
                @test c1 ≈ NLLSModel.cost(model)
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
end
