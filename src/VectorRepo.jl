
struct VectorRepo{T} # T is Any, or a Tuple of Types
    data::Dict{DataType, Vector}
    function VectorRepo{Any}()
        return new{Any}(Dict{DataType, Vector}())
    end
    function VectorRepo{T}() where T<:Tuple
        vr = new{T}(Dict{DataType, Vector}())
        foreach(t -> vr.data[t] = Vector{t}(), T.parameters)
        return vr
    end
    function VectorRepo(v::Vector{T}) where T
        vr = new{Tuple{T}}(Dict{DataType, Vector}())
        vr.data[T] = v
        return vr
    end
end

# Canonicalize any Union or single type to a Tuple type at compile time
tupelize(::Type{Union{}}) = Tuple{}
tupelize(::Type{T}) where T<:Tuple = T
tupelize(::Type{Any}) = Any
@generated function tupelize(::Type{T}) where T
    function extract(t)
        t isa Union ? [extract(t.a)..., extract(t.b)...] : [t]
    end
    types = extract(T)
    return :( Tuple{$(types...)} )
end

VectorRepo() = VectorRepo{Any}()
VectorRepo{T}() where T = VectorRepo{tupelize(T)}()

function Base.get(vr::VectorRepo{Any}, ::Type{type})::Vector{type} where type
    return haskey(vr.data, type) ? @inbounds(vr.data[type]::Vector{type}) : Vector{type}()
end

function Base.get(vr::VectorRepo{T}, ::Type{type})::Vector{type} where {T, type}
    @assert type in T.parameters "Invalid type"
    return @inbounds vr.data[type]::Vector{type}
end

function Base.get!(vr::VectorRepo{Any}, ::Type{type})::Vector{type} where type
    if haskey(vr.data, type)
        vec = @inbounds vr.data[type]::Vector{type}
    else
        vec = Vector{type}()
        vr.data[type] = vec
    end
    return vec
end

Base.get!(vr::VectorRepo{T}, ::Type{type}) where {T, type} = get(vr, type)
Base.push!(vr::VectorRepo, v::T) where T = push!(get!(vr, T), v)
Base.append!(vr::VectorRepo, v::Vector{T}) where T = append!(get!(vr, T), v)

# Get the keys
Base.keys(vr::VectorRepo{Any}) = keys(vr.data)
Base.keys(::VectorRepo{T}) where T = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))

# Get a vector of the vectors (allocates for Any, typed Tuple for T)
Base.values(vr::VectorRepo{Any}) = values(vr.data)
Base.values(vr::VectorRepo{T}) where T = ntuple(i -> get(vr, fieldtype(T, i)), Val(fieldcount(T)))
valuetuple(vr::VectorRepo, ::Type{T}) where T = values(vr)

# Sum reduction
Base.sum(fun, vr::VectorRepo; kw...) = sum(bindleadingargs(vrsum, fun, kw), values(vr); kw...)
vrsum(fun, kw, v::Vector) = sum(fun, v; kw...)

# Sum reduction over a subset of the elements
sumsubset(fun, vr::VectorRepo, subsetfun; init=0.0) = sum(bindleadingargs(sumsubsetvec, fun, subsetfun, init), values(vr); init=init)
sumsubsetvec(fun, subsetfun, init, vector) = sum(fun, subsetfun(vector); init=init)
