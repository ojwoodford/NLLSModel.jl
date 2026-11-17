# `fixture(names...)` includes each test/fixtures/<name>.jl into Main once per session, so that including a test file
# again defines no new type (and recompiles nothing that has not changed). Each test file includes this file and then
# names the fixtures it uses; each fixture file names those it uses itself. A second include of this file does nothing.
if !@isdefined(fixture)
    const LOADED_FIXTURES = Set{String}()
    function fixture(names::String...)
        for name in names
            name in LOADED_FIXTURES && continue
            Base.include(Main, joinpath(@__DIR__, name * ".jl"))
            push!(LOADED_FIXTURES, name)
        end
        return nothing
    end
end
