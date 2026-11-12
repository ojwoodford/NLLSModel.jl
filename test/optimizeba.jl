using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
using OptimizationVariableInterface, BlockLinearSystems
using OptimizationVariableInterface: nvars

# Simple affine projection transform
NLLSModel.generatemeasurement(pose::SVector{6, T}, X::SVector{3, U}) where {T, U} = SVector(dot(@inbounds(view(pose, NLLSModel.SR(1, 3))), X), dot(@inbounds(view(pose, NLLSModel.SR(4, 6))), X))

function create_ba_problem(ncameras, nlandmarks, propvisible)
    problem = Model(Union{SVector{6, Float64}, SVector{3, Float64}}, SimpleError2{2, Float64, SVector{6, Float64}, SVector{3, Float64}})

    # Generate the cameras on a unit sphere, pointing to the origin
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i = 1:ncameras
        addvariable!(problem, randn(SVector{6, Float64}) .+ camoffset)
    end
    
    # Generate the landmarks in a unit cube centered on the origin
    lmoffset = SVector(-0.5, -0.5, 10.0)
    for i = 1:nlandmarks
        addvariable!(problem, rand(SVector{3, Float64}) .+ lmoffset)
    end

    # Generate the measurements
    visibility = abs.(repeat(vec(1:ncameras), outer=(1, nlandmarks)) .- LinRange(2, ncameras-1, nlandmarks)')
    visibility = visibility .<= sort(vec(visibility))[Int(ceil(length(visibility)*propvisible))]
    for camind = 1:ncameras
        camera = problem.variables[camind]::SVector{6, Float64}
        for (landmark, tf) in enumerate(view(visibility, camind, :)')
            if tf
                landmarkind = landmark + ncameras
                addcost!(problem, SimpleError2{SVector{6, Float64}, SVector{3, Float64}}(generatemeasurement(camera, problem.variables[landmarkind]::SVector{3, Float64}), camind, landmarkind))
            end
        end
    end

    # Return the NLLSModel
    return problem
end

function perturb_ba_problem(problem, pointnoise, posenoise)
    for ind in 1:lastindex(problem.variables)
        if isa(problem.variables[ind], SVector{3, Float64})
            problem.variables[ind]::SVector{3, Float64} += randn(SVector{3, Float64}) * pointnoise
        else
            problem.variables[ind]::SVector{6, Float64} += randn(SVector{6, Float64}) * posenoise
        end
    end
    return problem
end

function compute_cost(problem)
    allocd = @allocated c = NLLSModel.cost(problem, static(1))
    return (c, allocd)
end

@testset "optimizeba.jl" begin
    # Generate some test data for a dense problem
    Random.seed!(1)
    problem = create_ba_problem(3, 5, 1.0)

    problem = perturb_ba_problem(problem, 0.003, 0.0)

    # Zero-allocation cost evaluation
    costbefore, allocd = compute_cost(problem)
    @test allocd == 0

    # Test optimvars on dense BA problem
    f_dense, fdf_dense!, x0_dense, H_dense = optimvars(problem, nothing, static(1))
    @test H_dense === nothing
    @test f_dense(x0_dense) ≈ costbefore
    dim_dense = nvars(x0_dense)
    g_dense = zeros(dim_dense)
    H_mat_dense = zeros(dim_dense, dim_dense)
    @test fdf_dense!(g_dense, H_mat_dense, x0_dense) ≈ costbefore
    @test isapprox(H_mat_dense, H_mat_dense'; atol=1e-14)

    # Test optimvars on sparse BA problem (zero-allocation f and fdf!)
    problem_sp = perturb_ba_problem(create_ba_problem(10, 50, 0.3), 0.001, 0.001)
    f_sp, fdf_sp!, x0_sp, H_sp = optimvars(problem_sp, nothing, static(1))
    @test H_sp isa SparseMatrixCSC{Float64, Int}
    dim_sp = nvars(x0_sp)
    @test size(H_sp) == (dim_sp, dim_sp)
    g_sp = zeros(dim_sp)
    c_sp = fdf_sp!(g_sp, H_sp, x0_sp)
    @test c_sp ≈ f_sp(x0_sp)
    @test isapprox(Matrix(H_sp), Matrix(H_sp)'; atol=1e-14)
    @test (@allocated f_sp(x0_sp)) == 0
    @test (@allocated fdf_sp!(g_sp, H_sp, x0_sp)) == 0

    # Test interleaved variable addition (Cam, Pt, Cam, Pt...)
    # Variables added in arbitrary order should be mapped to an RLE-structured linear system
    problem_interleaved = Model(Union{SVector{6, Float64}, SVector{3, Float64}}, SimpleError2{2, Float64, SVector{6, Float64}, SVector{3, Float64}})
    c1 = addvariable!(problem_interleaved, SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    p1 = addvariable!(problem_interleaved, SVector(0.1, 0.2, 5.0))
    c2 = addvariable!(problem_interleaved, SVector(1.1, 0.1, 0.0, 0.0, 1.0, 0.1))
    p2 = addvariable!(problem_interleaved, SVector(0.2, 0.3, 5.5))
    addcost!(problem_interleaved, SimpleError2{SVector{6, Float64}, SVector{3, Float64}}(NLLSModel.generatemeasurement(problem_interleaved.variables[c1], problem_interleaved.variables[p1]), c1, p1))
    addcost!(problem_interleaved, SimpleError2{SVector{6, Float64}, SVector{3, Float64}}(NLLSModel.generatemeasurement(problem_interleaved.variables[c2], problem_interleaved.variables[p2]), c2, p2))
    addcost!(problem_interleaved, SimpleError2{SVector{6, Float64}, SVector{3, Float64}}(NLLSModel.generatemeasurement(problem_interleaved.variables[c1], problem_interleaved.variables[p2]), c1, p2))

    f_int, fdf_int!, x0_int, _ = optimvars(problem_interleaved, nothing, static(1))
    # Verify that x0_int is grouped by block dimension: 2 cameras of size 6, then 2 points of size 3
    @test nvars(x0_int[1]) == 6
    @test nvars(x0_int[2]) == 6
    @test nvars(x0_int[3]) == 3
    @test nvars(x0_int[4]) == 3
    # Check that linear system internal block sizes form a BlockRLE with exactly 2 runs: [2 => 6, 2 => 3]
    orig_A_int = NLLSModel.getA(fdf_int!.linsystem)
    @test orig_A_int.rowblocksizes isa NLLSModel.BlockRLE
    @test NLLSModel.nruns(orig_A_int.rowblocksizes) == 2
    @test orig_A_int.rowblocksizes.runlengths == [2, 2]
    @test orig_A_int.rowblocksizes.blocksizes == [6, 3]

    # Evaluate cost and derivatives on interleaved problem
    cost_int = f_int(x0_int)
    @test cost_int >= 0
    dim_int = nvars(x0_int)
    g_int = zeros(dim_int)
    H_int = zeros(dim_int, dim_int)
    @test isapprox(fdf_int!(g_int, H_int, x0_int), cost_int; atol=1e-12)
    @test isapprox(H_int, H_int'; atol=1e-14)

    # --------------------------------------------------------------------------
    # Schur Complement Solver Integration Tests
    # --------------------------------------------------------------------------
    # 1. Custom Eliminable Landmark Type
    struct Landmark3D{T}
        p::SVector{3, T}
    end
    OptimizationVariableInterface.nvars(::Landmark3D) = static(3)
    OptimizationVariableInterface.update(p::Landmark3D, step, offset=1) = Landmark3D(p.p + SVector(step[offset], step[offset+1], step[offset+2]))
    OptimizationVariableInterface.iseliminable(::Type{<:Landmark3D}) = static(true)
    NLLSModel.generatemeasurement(camera, pt::Landmark3D) = NLLSModel.generatemeasurement(camera, pt.p)

    # 2. Build BA problem with Landmark3D
    ncams = 4
    nlms = 8
    prob_schur = Model(Union{SVector{6, Float64}, Landmark3D{Float64}}, SimpleError2{2, Float64, SVector{6, Float64}, Landmark3D{Float64}})
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i in 1:ncams
        addvariable!(prob_schur, randn(SVector{6, Float64}) .+ camoffset)
    end
    lmoffset = SVector(-0.5, -0.5, 10.0)
    for i in 1:nlms
        addvariable!(prob_schur, Landmark3D(rand(SVector{3, Float64}) .+ lmoffset))
    end
    for c in 1:ncams
        camera = prob_schur.variables[c]::SVector{6, Float64}
        for l in 1:nlms
            if (c + l) % 2 == 0 || c == 1
                l_idx = ncams + l
                addcost!(prob_schur, SimpleError2{SVector{6, Float64}, Landmark3D{Float64}}(NLLSModel.generatemeasurement(camera, prob_schur.variables[l_idx]), c, l_idx))
            end
        end
    end

    # 3. Test Bipartite Validation
    # A) Connecting 2 eliminable variables throws ArgumentError
    NLLSModel.generatemeasurement(pt1::Landmark3D, pt2::Landmark3D) = SVector(pt1.p[1] - pt2.p[1], pt1.p[2] - pt2.p[2])
    prob_bad = Model(Union{SVector{6, Float64}, Landmark3D{Float64}}, Union{SimpleError2{2, Float64, SVector{6, Float64}, Landmark3D{Float64}}, SimpleError2{2, Float64, Landmark3D{Float64}, Landmark3D{Float64}}})
    addvariable!(prob_bad, randn(SVector{6, Float64}))
    addvariable!(prob_bad, Landmark3D(SVector(1.0, 2.0, 3.0)))
    addvariable!(prob_bad, Landmark3D(SVector(4.0, 5.0, 6.0)))
    addcost!(prob_bad, SimpleError2{SVector{6, Float64}, Landmark3D{Float64}}(SVector(0.0, 0.0), 1, 2))
    addcost!(prob_bad, SimpleError2{Landmark3D{Float64}, Landmark3D{Float64}}(SVector(0.0, 0.0), 2, 3))
    @test_throws ArgumentError partition_schur_problem(prob_bad, trues(length(prob_bad.variables)), :schur)

    # B) Calling with no eliminable variables when linsolver=:schur throws ArgumentError
    prob_no_elim = create_ba_problem(2, 2, 1.0)
    @test_throws ArgumentError partition_schur_problem(prob_no_elim, trues(length(prob_no_elim.variables)), :schur)

    # 4. Dense Schur Solver (optimvars)
    f_ds, fdf_ds!, x0_ds, H_ds = optimvars(prob_schur, nothing, static(1); linsolver=:dense_schur)
    @test H_ds isa BlockLinearSystems.SchurBlockMatrix
    @test H_ds.B isa Matrix{Float64}
    tot_dim_ds = H_ds.structure.nc + sum(H_ds.structure.p_dims)
    g_ds = zeros(tot_dim_ds)
    c_ds = fdf_ds!(g_ds, H_ds, x0_ds)
    @test isapprox(c_ds, f_ds(x0_ds); atol=1e-12)
    @test (@allocated fdf_ds!(g_ds, H_ds, x0_ds)) == 0

    # 5. Sparse Schur Solver (optimvars)
    f_ss, fdf_ss!, x0_ss, H_ss = optimvars(prob_schur, nothing, static(1); linsolver=:sparse_schur)
    @test H_ss isa BlockLinearSystems.SchurBlockMatrix
    @test H_ss.B isa BlockLinearSystems.SparseBlockLinearSystem
    g_ss = zeros(tot_dim_ds)
    c_ss = fdf_ss!(g_ss, H_ss, x0_ss)
    @test isapprox(c_ss, f_ss(x0_ss); atol=1e-12)
    @test (@allocated fdf_ss!(g_ss, H_ss, x0_ss)) == 0

    # Verify dense and sparse Schur Hessians match on quadratic form
    dx_rand = randn(tot_dim_ds)
    val_dense = fast_bAb(H_ds, dx_rand)
    val_sparse = fast_bAb(H_ss, dx_rand)
    @test isapprox(val_dense, val_sparse; rtol=1e-12, atol=1e-12)

    # 5b. Retained-retained cost with no shared eliminated variable (B1): sparse and dense Schur agree
    NLLSModel.generatemeasurement(p1::SVector{6, T}, p2::SVector{6, U}) where {T, U} = SVector(p1[1] - p2[1], p1[4] - p2[5])
    CT = Union{SimpleError2{2, Float64, SVector{6, Float64}, Landmark3D{Float64}}, SimpleError2{2, Float64, SVector{6, Float64}, SVector{6, Float64}}}
    prob_rr = Model(Union{SVector{6, Float64}, Landmark3D{Float64}}, CT)
    for i in 1:3
        addvariable!(prob_rr, randn(SVector{6, Float64}) .+ camoffset)
    end
    for i in 1:4
        addvariable!(prob_rr, Landmark3D(rand(SVector{3, Float64}) .+ lmoffset))
    end
    for cam in 1:2, l in 4:7 # landmarks are seen by poses 1 and 2 only
        addcost!(prob_rr, SimpleError2{SVector{6, Float64}, Landmark3D{Float64}}(randn(SVector{2, Float64}), cam, l))
    end
    addcost!(prob_rr, SimpleError2{SVector{6, Float64}, SVector{6, Float64}}(randn(SVector{2, Float64}), 2, 3)) # relative-pose cost
    @test partition_schur_problem(prob_rr, trues(7), :sparse_schur).extra_blocks == [(2, 3)]
    f_rd, fdf_rd!, x0_rd, H_rd = optimvars(prob_rr, nothing, static(1); linsolver=:dense_schur)
    f_rs, fdf_rs!, x0_rs, H_rs = optimvars(prob_rr, nothing, static(1); linsolver=:sparse_schur)
    tot_rr = H_rd.structure.nc + sum(H_rd.structure.p_dims)
    g_rd = zeros(tot_rr); g_rs = zeros(tot_rr)
    @test fdf_rd!(g_rd, H_rd, x0_rd) ≈ fdf_rs!(g_rs, H_rs, x0_rs)
    @test g_rd ≈ g_rs
    dx_rr = randn(tot_rr)
    @test fast_bAb(H_rd, dx_rr) ≈ fast_bAb(H_rs, dx_rr)

    # 6. Test CostSlotMap Construction
    num_costs = NLLSModel.countcosts(NLLSModel.costnum, prob_schur.costs)
    slot_map_dense = CostSlotMap(prob_schur, H_ds.structure, fdf_ds!.linsystem.x.blockindices, H_ds.B)
    @test length(slot_map_dense.c_offsets) == num_costs
    @test length(slot_map_dense.b_ptr) == num_costs + 1
    @test length(slot_map_dense.e_ptr) == num_costs + 1

    slot_map_sparse = CostSlotMap(prob_schur, H_ss.structure, fdf_ss!.linsystem.x.blockindices, H_ss.B)
    @test length(slot_map_sparse.c_offsets) == num_costs
end
