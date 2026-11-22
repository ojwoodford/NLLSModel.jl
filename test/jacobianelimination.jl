using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using SecondOrderOptimization, NLPModels
import BlockLinearSystems
import BlockLinearSystems: SchurBlockJacobian, block, blockentry, residual_block, copy_structure
import OptimizationVariableInterface

include("fixtures/loader.jl")
fixture("allocations", "frontdoor", "jacobianelimination")

# NTP2: elimination on the Jacobian path, by QR (a SchurBlockJacobian prototype).

@testset "jacobianelimination.jl" begin
    @testset "the prototype: $name" for (name, make, unfixed) in ELIM_MODELS
        model = make()
        J(; kw...) = optimvars_nls(model; unfixed, numthreads=static(1), kw...)[5]
        @test J(reduction=:eliminate, storage=:dense) isa SchurBlockJacobian{Float64, Matrix{Float64}}
        @test J(reduction=:eliminate, storage=:sparse) isa SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}
        @test J() isa SchurBlockJacobian # :auto eliminates: the model has eliminable variables and the structure fits
        @test J(reduction=:none) isa Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int}}
        # The reduced storage follows the Hessian path's rule: the same choice for the same model
        Hauto = optimvars(model; unfixed, numthreads=static(1))[4]
        @test (J() isa SchurBlockJacobian{Float64, Matrix{Float64}}) == (Hauto.R isa Matrix)
    end

    @testset "the same cost as the Hessian path: $name, $storage" for (name, make, unfixed) in ELIM_MODELS, storage in (:dense, :sparse)
        model = make()
        c0 = NLLSModel.cost(model)
        solved_hes, _ = optimize(model, ElimOptions; unfixed, numthreads=static(1))
        solved_jac, rt = optimize(model, ElimOptions; formulation=:jacobian, reduction=:eliminate, storage, unfixed, numthreads=static(1))
        plain, _ = optimize(model, ElimOptions; formulation=:jacobian, reduction=:none, unfixed, numthreads=static(1))
        @test NLLSModel.cost(solved_jac) ≈ NLLSModel.cost(solved_hes) rtol=1e-8 atol=1e-10 * c0
        @test NLLSModel.cost(solved_jac) ≈ NLLSModel.cost(plain) rtol=1e-8 atol=1e-10 * c0
        # linsolver = :qr, a QR of the reduced Jacobian, dense or sparse (a sparse one since SecondOrderOptimization TP6)
        jacobianqr, _ = optimize(model, OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:qr, maxiters=200); formulation=:jacobian,
            reduction=:eliminate, storage, unfixed, numthreads=static(1))
        @test NLLSModel.cost(jacobianqr) ≈ NLLSModel.cost(solved_hes) rtol=1e-8 atol=1e-10 * c0
        # The solution is written back through the partition (syncvars!)
        @test NLLSModel.cost(solved_jac) ≈ rt.bestcost rtol=1e-12 atol=1e-10 * c0
        @test NLLSModel.cost(solved_jac) < 0.1 * c0
        # Only the unfixed variables moved
        unfixed === nothing || @test all(k -> solved_jac.variables[k] == model.variables[k], setdiff(eachindex(model.variables), unfixed))
    end

    @testset "the tables reproduce the plain Jacobian row for row: $name" for (name, make, unfixed) in ELIM_MODELS
        model = make()
        _, resp!, rjp, xp, Jp = optimvars_nls(model; unfixed, reduction=:none, storage=:dense, numthreads=static(1))
        _, rese!, rje, xe, Je = optimvars_nls(model; unfixed, reduction=:eliminate, numthreads=static(1))
        rp = zeros(size(Jp, 1))
        cp = rjp(rp, Jp, xp)
        fill!(Je.data, 7.0); fill!(Je.r, 7.0) # Not prezeroed: every entry is set
        ce = rje(Float64[], Je, xe) # The residual vector is not used
        @test ce == cp
        Jd, rd = tables_as_plain(rje, Je, rjp)
        @test Jd == Jp
        @test rd == rp
        # And nothing else is written: the damping rows and the entries no cost touches are zero
        # (the residual is in its own vector, Je.r, since BlockLinearSystems SP-B5)
        @test sum(abs2, Je.data) + sum(abs2, Je.r) ≈ sum(abs2, Jp) + sum(abs2, rp) rtol=1e-14
        # With prezeroed = true the zeros are taken to be in place
        Jz = copy_structure(Je)
        @test rje(Float64[], Jz, xe; prezeroed=true) == ce && Jz.data == Je.data && Jz.r == Je.r
        # res! still writes the whole residual, in the plain order
        r2, r3 = zeros(length(rp)), zeros(length(rp))
        @test rese!(r2, xe) == resp!(r3, xp)
        @test r2 == r3
    end

    @testset "a Jacobian of another structure is refused" begin
        model = optlandmark_problem(4, 20)
        _, _, rj, x0, J = optimvars_nls(model; reduction=:eliminate, numthreads=static(1))
        J2 = optimvars_nls(model; reduction=:eliminate, numthreads=static(1))[5]
        @test_throws ArgumentError rj(Float64[], J2, x0)
        @test rj(Float64[], copy_structure(J), x0) ≈ NLLSModel.cost(model) rtol=1e-12
    end

    @testset "reduction = :auto and :eliminate on the Jacobian path" begin
        ba = optlandmark_problem(4, 20)
        plain, _ = chainproblem(5, 2, 9) # No eliminable variables
        @test optimvars_nls(ba; numthreads=static(1))[5] isa SchurBlockJacobian
        @test optimvars_nls(plain; numthreads=static(1))[5] isa Matrix{Float64}
        @test_throws ArgumentError optimvars_nls(plain; numthreads=static(1), reduction=:eliminate)
        # An eliminable variable in no cost: no rows to eliminate, so :auto does not eliminate and :eliminate throws
        lonely = optlandmark_problem(4, 20)
        addvariable!(lonely, OptLandmark{Float64}(SVector(0.0, 0.0, 10.0)))
        @test (@test_logs (:warn,) match_mode=:any optimvars_nls(lonely; numthreads=static(1))[5]) isa Matrix{Float64}
        err = try optimvars_nls(lonely; numthreads=static(1), reduction=:eliminate); nothing catch e e end
        @test err isa ArgumentError && occursin("in none", err.msg)
        # Front door: :eliminate is not rejected
        @test optimize!(deepcopy(ba), ElimOptions; formulation=:jacobian, reduction=:eliminate, numthreads=static(1)) isa OptimizationResult
        # A single unfixed variable is never partitioned, as on the Hessian path
        @test optimvars_nls(ba; unfixed=1, numthreads=static(1), reduction=:eliminate)[5] isa Matrix{Float64}
        # A dense reduced system above the cap: an ArgumentError naming it (4 cameras of 6: 24×24, 4608 bytes)
        err = try optimvars_nls(ba; numthreads=static(1), reduction=:eliminate, storage=:dense, max_dense_bytes=4607); nothing catch e e end
        @test err isa ArgumentError && occursin("24×24 reduced system", err.msg)
        @test optimvars_nls(ba; numthreads=static(1), reduction=:eliminate, max_dense_bytes=4607)[5] isa SchurBlockJacobian{Float64, SparseMatrixCSC{Float64, Int}}
    end

    @testset "the wrappers that hand a plain Jacobian on never eliminate" begin
        ba = flatelimmodel()
        @test optimvars_nls(ba; numthreads=static(1))[5] isa SchurBlockJacobian # Elimination is available on the Jacobian path
        for (name, entry) in (("flat_optimvars_nls", flat_optimvars_nls), ("nls_callbacks", NLLSModel.nls_callbacks), ("NLLSNLSModel", NLLSNLSModel))
            err = try entry(ba; numthreads=static(1), reduction=:eliminate); nothing catch e e end
            @test err isa ArgumentError && occursin(name, err.msg) && occursin("plain Jacobian", err.msg)
        end
        @test flat_optimvars_nls(ba; numthreads=static(1)).J_proto isa Matrix{Float64} # :auto does not eliminate here
        @test NLLSModel.nls_callbacks(ba; numthreads=static(1)).J0 isa Matrix{Float64}
        @test NLLSNLSModel(ba; numthreads=static(1)).J_buf isa Matrix{Float64}
    end
end
