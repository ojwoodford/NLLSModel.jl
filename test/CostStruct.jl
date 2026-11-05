using NLLSModel, Test

function fillcoststruct(cs, floats, ints)
    # Fill the CostStruct with the data
    Nf = length(floats)
    Ni = length(ints)
    append!(cs, floats) 
    append!(cs, ints)
    push!(cs, 0.0)
    push!(cs, 0)

    # Test that the lengths and values are as expected
    floats_out = get(cs, Float64)
    @test length(floats_out) == Nf+1 && floats_out[end] == 0
    @test cs[Float64] === floats_out && get!(cs, Float64) === floats_out
    ints_out = get(cs, Int)
    @test length(ints_out) == Ni+1 && ints_out[end] == 0
    @test cs[Int] === ints_out && get!(cs, Int) === ints_out
end

@testset "CostStruct.jl" begin
    # Generate random data
    floats = rand(10) * 100
    ints = convert(Vector{Int}, ceil.(floats))
    total = sum(floats) + sum(ints)
    halftotal = sum(floats[1:5]) + sum(ints[1:5])
    rangefun(v::Vector{T}) where T = T <: Char ? T[] : view(v, 1:5)
    indicesfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, [1, 2, 3, 4, 5])
    bitvecfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, 1:11 .<= 5)
    boolvecfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, map(x->x<=5, 1:11))

    # Construct CostStructs and test reductions
    # Union container (converts to Tuple parameter)
    cs2 = NLLSModel.CostStruct{Union{Float64, Int, Char}}()
    @test isa(cs2, NLLSModel.CostStruct{<:Tuple})
    keytup = keys(cs2)
    @test isa(keytup, Tuple) && length(keytup) == 3
    @test any(keytup .== Float64) && any(keytup .== Int) && any(keytup .== Char)
    @test haskey(cs2, Float64) && haskey(cs2, Int) && haskey(cs2, Char) && !haskey(cs2, String)
    @test_throws ErrorException get(cs2, String)
    @test_throws ErrorException cs2[String]
    @test_throws ErrorException push!(cs2, "invalid")
    @test_throws ArgumentError sum(i->2i, cs2)
    @test sum(i->2i, cs2; init=0.0) == 0.0
    @test sum(i->2i, cs2; init=5.0) == 5.0
    fillcoststruct(cs2, floats, ints)
    @test sum(i->π*i, cs2; init=0.0) ≈ total * π
    @test sum(i->π*i, cs2; init=10.0) ≈ total * π + 10.0
    @test sum(i->π*i, cs2) ≈ total * π
    valuetup = values(cs2)
    @test isa(valuetup, Tuple) && length(valuetup) == 3
    @test any(Base.Fix2(isa, Vector{Float64}), valuetup) && any(Base.Fix2(isa, Vector{Int}), valuetup) && any(Base.Fix2(isa, Vector{Char}), valuetup)

    # Test subset reductions
    @test NLLSModel.sumsubset(Float64, cs2, rangefun) == halftotal
    @test NLLSModel.sumsubset(Float64, cs2, rangefun; init=7.0) == halftotal + 7.0
    @test NLLSModel.sumsubset(Float64, cs2, indicesfun) == halftotal
    @test NLLSModel.sumsubset(Float64, cs2, bitvecfun) == halftotal
    @test NLLSModel.sumsubset(Float64, cs2, boolvecfun) ≈ halftotal

    # Tuple container
    cs2_tup = NLLSModel.CostStruct{Tuple{Float64, Int, Char}}()
    @test isa(cs2_tup, NLLSModel.CostStruct{Tuple{Float64, Int, Char}})
    keytup = keys(cs2_tup)
    @test isa(keytup, Tuple) && length(keytup) == 3
    @test any(keytup .== Float64) && any(keytup .== Int) && any(keytup .== Char)
    fillcoststruct(cs2_tup, floats, ints)
    @test sum(i->π*i, cs2_tup; init=0.0) ≈ total * π
    valuetup = values(cs2_tup)
    @test isa(valuetup, Tuple) && length(valuetup) == 3
    @test any(Base.Fix2(isa, Vector{Float64}), valuetup) && any(Base.Fix2(isa, Vector{Int}), valuetup) && any(Base.Fix2(isa, Vector{Char}), valuetup)
    @test NLLSModel.sumsubset(Float64, cs2_tup, rangefun) == halftotal

    # Initialize from single vector
    cs3 = NLLSModel.CostStruct([1, 2])
    @test isa(cs3, NLLSModel.CostStruct{Tuple{Int}})
    @test sum(i->i, cs3) == 3
    @test sum(cs3) == 3
    @test NLLSModel.sumsubset(Float64, cs3, v -> view(v, 1:1)) == 1.0
    keytup = keys(cs3)
    @test isa(keytup, Tuple) && length(keytup) == 1 && keytup[1] == Int

    # Single type container
    cs5 = NLLSModel.CostStruct{Float64}()
    @test isa(cs5, NLLSModel.CostStruct{Tuple{Float64}})
    push!(cs5, 3.14)
    @test sum(identity, cs5) == 3.14
    @test sum(cs5) == 3.14
    @test keys(cs5) == (Float64,)
    @test values(cs5) == ([3.14],)

    # Empty Union{} container & NLLSModel cost type validation
    cs_empty = NLLSModel.CostStruct{Union{}}()
    @test isa(cs_empty, NLLSModel.CostStruct{Tuple{}})
    @test keys(cs_empty) == ()
    @test values(cs_empty) == ()
    @test sum(identity, cs_empty; init=0.0) == 0.0
    @test_throws ErrorException get(cs_empty, Float64)
    @test_throws ArgumentError NLLSModel{Float64, Int}(Float64[], cs5)
end
