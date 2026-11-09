using NLLSModel, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
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

    # Test reordering the costs
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    costbefore = cost(problem)
    landmarks = isa.(problem.variables, SVector{3, Float64})
    NLLSModel.reordercostsforschur!(problem, landmarks)
    @test cost(problem) ≈ costbefore

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
end
