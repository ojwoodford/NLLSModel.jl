using NLLSModels, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra

# Simple affine projection transform
NLLSModels.generatemeasurement(pose::EuclideanVector{6, T}, X::EuclideanVector{3, U}) where {T, U} = SVector(dot(@inbounds(view(pose, NLLSModels.SR(1, 3))), X), dot(@inbounds(view(pose, NLLSModels.SR(4, 6))), X))

function create_ba_problem(ncameras, nlandmarks, propvisible)
    problem = NLLSProblem(Union{EuclideanVector{6, Float64}, EuclideanVector{3, Float64}}, SimpleError2{2, Float64, EuclideanVector{6, Float64}, EuclideanVector{3, Float64}})

    # Generate the cameras on a unit sphere, pointing to the origin
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i = 1:ncameras
        addvariable!(problem, randn(EuclideanVector{6, Float64}) .+ camoffset)
    end
    
    # Generate the landmarks in a unit cube centered on the origin
    lmoffset = SVector(-0.5, -0.5, 10.0)
    for i = 1:nlandmarks
        addvariable!(problem, rand(EuclideanVector{3, Float64}) .+ lmoffset)
    end

    # Generate the measurements
    visibility = abs.(repeat(vec(1:ncameras), outer=(1, nlandmarks)) .- LinRange(2, ncameras-1, nlandmarks)')
    visibility = visibility .<= sort(vec(visibility))[Int(ceil(length(visibility)*propvisible))]
    for camind = 1:ncameras
        camera = problem.variables[camind]::EuclideanVector{6, Float64}
        for (landmark, tf) in enumerate(view(visibility, camind, :)')
            if tf
                landmarkind = landmark + ncameras
                addcost!(problem, SimpleError2{EuclideanVector{6, Float64}, EuclideanVector{3, Float64}}(generatemeasurement(camera, problem.variables[landmarkind]::EuclideanVector{3, Float64}), camind, landmarkind))
            end
        end
    end

    # Return the NLLSProblem
    return problem
end

function perturb_ba_problem(problem, pointnoise, posenoise)
    for ind in 1:lastindex(problem.variables)
        if isa(problem.variables[ind], EuclideanVector{3, Float64})
            problem.variables[ind]::EuclideanVector{3, Float64} += randn(SVector{3, Float64}) * pointnoise
        else
            problem.variables[ind]::EuclideanVector{6, Float64} += randn(SVector{6, Float64}) * posenoise
        end
    end
    return problem
end

function compute_cost(problem)
    allocd = @allocated c = NLLSModels.cost(problem, static(1))
    return (c, allocd)
end

@testset "optimizeba.jl" begin
    # Generate some test data for a dense problem
    Random.seed!(1)
    problem = create_ba_problem(3, 5, 1.0)

    # Test reordering the costs
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    costbefore = cost(problem)
    landmarks = isa.(problem.variables, EuclideanVector{3, Float64})
    NLLSModels.reordercostsforschur!(problem, landmarks)
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
end
