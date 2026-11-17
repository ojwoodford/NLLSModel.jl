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
