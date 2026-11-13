using NLLSModel, Test, Static, StaticArrays, SparseArrays, LinearAlgebra
import SecondOrderOptimization: optimize!, OptimizationOptions, HessianObjective
import OptimizationVariableInterface

# The 255-dimension limit comes from BlockLinearSystems' UInt8 block sizes. It applies only to variables that become blocks
# of a block linear system, not to variables optimized on their own or to the residual-and-Jacobian path.

# Pins a variable of any size to a measurement
struct PinRes <: NLLSModel.AbstractResidual
    meas::Vector{Float64}
    i::Int
end
NLLSModel.ndeps(::PinRes) = static(1)
NLLSModel.nres(r::PinRes) = length(r.meas)
NLLSModel.varindices(r::PinRes) = r.i
NLLSModel.getvars(r::PinRes, vars::AbstractVector) = (vars[r.i]::Vector{Float64},)
NLLSModel.computeresidual(r::PinRes, x) = x .- r.meas
Base.eltype(::PinRes) = Float64

# Links the first three entries of one variable to another variable of three dimensions
struct LinkRes <: NLLSModel.AbstractResidual
    i::Int
    j::Int
end
NLLSModel.ndeps(::LinkRes) = static(2)
NLLSModel.nres(::LinkRes) = static(3)
NLLSModel.varindices(r::LinkRes) = SVector(r.i, r.j)
NLLSModel.getvars(r::LinkRes, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::Vector{Float64})
NLLSModel.computeresidual(::LinkRes, a, b) = SVector(a[1] - b[1], a[2] - b[2], a[3] - b[3])
Base.eltype(::LinkRes) = Float64

# Eliminable three-dimensional variable, for the Schur systems
struct SmallLandmark
    p::SVector{3, Float64}
end
OptimizationVariableInterface.nvars(::SmallLandmark) = static(3)
OptimizationVariableInterface.update(v::SmallLandmark, step, offset=1) = SmallLandmark(v.p + SVector(step[offset], step[offset+1], step[offset+2]))
OptimizationVariableInterface.iseliminable(::Type{SmallLandmark}) = static(true)

struct LandmarkRes <: NLLSModel.AbstractResidual
    i::Int
    j::Int
end
NLLSModel.ndeps(::LandmarkRes) = static(2)
NLLSModel.nres(::LandmarkRes) = static(3)
NLLSModel.varindices(r::LandmarkRes) = SVector(r.i, r.j)
NLLSModel.getvars(r::LandmarkRes, vars::AbstractVector) = (vars[r.i]::Vector{Float64}, vars[r.j]::SmallLandmark)
NLLSModel.computeresidual(::LandmarkRes, a, l) = SVector(a[1] - l.p[1], a[2] - l.p[2], a[3] - l.p[3])
Base.eltype(::LandmarkRes) = Float64

function schurproblem(n)
    problem = Model(Union{Vector{Float64}, SmallLandmark}, LandmarkRes)
    addvariable!(problem, zeros(n))
    for _ in 1:3
        addvariable!(problem, SmallLandmark(SVector(1.0, 2.0, 3.0)))
    end
    for j in 2:4
        addcost!(problem, LandmarkRes(1, j))
    end
    return problem
end

const CostUnion = Union{PinRes, LinkRes}

# Variable 1 has `n` dimensions; variables 2 to `nsmall + 1` have three
function bigproblem(n, nsmall=3)
    problem = Model(Vector{Float64}, CostUnion)
    addvariable!(problem, zeros(n))
    for _ in 1:nsmall
        addvariable!(problem, [1.0, 2.0, 3.0])
    end
    addcost!(problem, PinRes(collect(1.0:n), 1))
    for j in 2:nsmall+1
        addcost!(problem, PinRes([1.0, 2.0, 3.0], j))
        addcost!(problem, LinkRes(1, j))
    end
    return problem
end

@testset "bigvariable.jl" begin
    # A variable of any size can be added
    problem = bigproblem(256)
    @test length(problem.variables[1]) == 256

    # It can be optimized on its own, and the single variable system does not use blocks
    f, fdf!, x0, H = optimvars(problem, 1, static(1))
    @test H === nothing
    res = optimize!(x0, HessianObjective(f, fdf!; H), OptimizationOptions(iterator=:levenbergmarquardt, maxiters=20))
    x = x0
    @test f(x) < 1e-8
    @test x ≈ collect(1.0:256)

    # The residual-and-Jacobian path writes a flat Jacobian, so it accepts a larger variable too
    f, res!, resjac!, x0, J = optimvars_nls(problem; unfixed=1, numthreads=static(1))
    r = zeros(size(J, 1))
    Jm = copy(J)
    @test resjac!(r, Jm, x0) ≈ f(x0)
    @test size(J, 2) == 256

    # Together with other variables, every block linear system rejects it with an informative error
    function checkerror(thunk)
        err = try thunk(); nothing catch e e end
        @test err isa ArgumentError
        msg = err isa ArgumentError ? err.msg : ""
        @test occursin("Variable 1", msg) && occursin("256", msg) && occursin("1 to 255", msg) && occursin("on its own", msg)
    end
    checkerror(() -> optimvars(problem, nothing, static(1); reduction=:none)) # Dense
    checkerror(() -> optimvars(bigproblem(256, 40), nothing, static(1); reduction=:none)) # Sparse (large enough to be chosen)
    for reduced in (:dense, :sparse, :auto) # Schur (the default chooses it for eliminable variables)
        checkerror(() -> optimvars(schurproblem(256), nothing, static(1); reduced))
    end

    # A variable of 255 dimensions still works as a block, including in a Schur system
    f, fdf!, x0, H = optimvars(schurproblem(255), nothing, static(1); reduced=:dense)
    @test f(x0) > 0
    problem = bigproblem(255)
    f, fdf!, x0, H = optimvars(problem, nothing, static(1))
    c0 = f(x0)
    optimize!(x0, HessianObjective(f, fdf!; H), OptimizationOptions(iterator=:levenbergmarquardt, maxiters=50))
    @test f(x0) < c0
end
