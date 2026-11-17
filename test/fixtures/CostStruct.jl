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
