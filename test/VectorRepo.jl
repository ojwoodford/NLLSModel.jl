using NLLSModels, Test

function fillrepo(vr, floats, ints)
    # Fill the repo with the data
    Nf = length(floats)
    Ni = length(ints)
    append!(vr, floats) 
    append!(vr, ints)
    push!(vr, 0.0)
    push!(vr, 0)

    # Test that the lengths and values are as expected
    floats = get(vr, Float64)
    @test length(floats) == Nf+1 && floats[end] == 0
    ints = get(vr, Int)
    @test length(ints) == Ni+1 && ints[end] == 0
end

@testset "VectorRepo.jl" begin
    # Generate random data
    floats = rand(10) * 100
    ints = convert(Vector{Int}, ceil.(floats))
    total = sum(floats) + sum(ints)
    halftotal = sum(floats[1:5]) + sum(ints[1:5])
    rangefun(v::Vector{T}) where T = T <: Char ? T[] : view(v, 1:5)
    indicesfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, [1, 2, 3, 4, 5])
    bitvecfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, 1:11 .<= 5)
    boolvecfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, map(x->x<=5, 1:11))

    # Construct repos and test the sum reduction
    # Any container
    vr1 = NLLSModels.VectorRepo()
    @test sum(i->2i, vr1; init=0.0) == 0.0
    fillrepo(vr1, floats, ints)
    @test sum(i->π*i, vr1; init=0.0) ≈ total * π
    vec = values(vr1)
    @test length(vec) == 2 && any(Base.Fix2(isa, Vector{Float64}), vec) && any(Base.Fix2(isa, Vector{Int}), vec)

    # Test subset reductions
    @test NLLSModels.sumsubset(Float64, vr1, rangefun) == halftotal
    @test NLLSModels.sumsubset(Float64, vr1, indicesfun) == halftotal
    @test NLLSModels.sumsubset(Float64, vr1, bitvecfun) == halftotal
    @test NLLSModels.sumsubset(Float64, vr1, boolvecfun) ≈ halftotal

    # Union container (converts to Tuple parameter)
    vr2 = NLLSModels.VectorRepo{Union{Float64, Int, Char}}()
    @test isa(vr2, NLLSModels.VectorRepo{<:Tuple})
    keytup = keys(vr2)
    @test isa(keytup, Tuple) && length(keytup) == 3
    @test any(keytup .== Float64) && any(keytup .== Int) && any(keytup .== Char)
    @test sum(i->2i, vr2; init=0.0) == 0.0
    fillrepo(vr2, floats, ints)
    @test sum(i->π*i, vr2; init=0.0) ≈ total * π
    valuetup = values(vr2)
    @test isa(valuetup, Tuple) && length(valuetup) == 3
    @test any(Base.Fix2(isa, Vector{Float64}), valuetup) && any(Base.Fix2(isa, Vector{Int}), valuetup) && any(Base.Fix2(isa, Vector{Char}), valuetup)

    # Test subset reductions
    @test NLLSModels.sumsubset(Float64, vr2, rangefun) == halftotal
    @test NLLSModels.sumsubset(Float64, vr2, indicesfun) == halftotal
    @test NLLSModels.sumsubset(Float64, vr2, bitvecfun) == halftotal
    @test NLLSModels.sumsubset(Float64, vr2, boolvecfun) ≈ halftotal
    @test keys(vr1) == keys(vr1.data)

    # Tuple container
    vr2_tup = NLLSModels.VectorRepo{Tuple{Float64, Int, Char}}()
    @test isa(vr2_tup, NLLSModels.VectorRepo{Tuple{Float64, Int, Char}})
    keytup = keys(vr2_tup)
    @test isa(keytup, Tuple) && length(keytup) == 3
    @test any(keytup .== Float64) && any(keytup .== Int) && any(keytup .== Char)
    fillrepo(vr2_tup, floats, ints)
    @test sum(i->π*i, vr2_tup; init=0.0) ≈ total * π
    valuetup = values(vr2_tup)
    @test isa(valuetup, Tuple) && length(valuetup) == 3
    @test any(Base.Fix2(isa, Vector{Float64}), valuetup) && any(Base.Fix2(isa, Vector{Int}), valuetup) && any(Base.Fix2(isa, Vector{Char}), valuetup)
    @test NLLSModels.sumsubset(Float64, vr2_tup, rangefun) == halftotal

    # Initialize from single vector
    vr3 = NLLSModels.VectorRepo([1, 2])
    @test isa(vr3, NLLSModels.VectorRepo{Tuple{Int}})
    @test sum(i->i, vr3) == 3
    @test NLLSModels.sumsubset(Float64, vr3, v -> view(v, 1:1)) == 1.0
    keytup = keys(vr3)
    @test isa(keytup, Tuple) && length(keytup) == 1 && keytup[1] == Int

    # Single type container
    vr5 = NLLSModels.VectorRepo{Float64}()
    @test isa(vr5, NLLSModels.VectorRepo{Tuple{Float64}})
    push!(vr5, 3.14)
    @test sum(identity, vr5) == 3.14
    @test keys(vr5) == (Float64,)
    @test values(vr5) == ([3.14],)
end
