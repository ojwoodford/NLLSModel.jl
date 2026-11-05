
# Canonicalize any Union or single type to a Tuple type at compile time
tupelize(::Type{Union{}}) = Tuple{}
tupelize(::Type{T}) where T<:Tuple = T
@generated function tupelize(::Type{T}) where T
    function extract(t)
        t isa Union ? [extract(t.a)..., extract(t.b)...] : [t]
    end
    types = extract(T)
    return :( Tuple{$(types...)} )
end

# Compile-time index of a type within a Tuple of types
@inline findtypeindex(::Type{target}, ::Type{target}, rest...) where target = 1
@inline findtypeindex(::Type{target}, ::Type{other}, rest...) where {target, other} = 1 + findtypeindex(target, rest...)
@inline findtypeindex(::Type{target}) where target = 0
@inline typeindex(::Type{T}, ::Type{target}) where {T<:Tuple, target} = findtypeindex(target, fieldtypes(T)...)

struct CostStruct{T, V<:Tuple}
    data::V
    function CostStruct{T}() where T<:Tuple
        vecs = ntuple(i -> Vector{fieldtype(T, i)}(), Val(fieldcount(T)))
        return new{T, typeof(vecs)}(vecs)
    end
    function CostStruct{T, V}(data::V) where {T<:Tuple, V<:Tuple}
        return new{T, V}(data)
    end
end

CostStruct{T}() where T = CostStruct{tupelize(T)}()
CostStruct(v::Vector{T}) where T = CostStruct{Tuple{T}, Tuple{Vector{T}}}((v,))

@inline function Base.get(vr::CostStruct{T}, ::Type{type})::Vector{type} where {T, type}
    idx = typeindex(T, type)
    @boundscheck idx > 0 || error("Type $type not in CostStruct{$T}")
    return @inbounds vr.data[idx]::Vector{type}
end

@inline Base.get!(vr::CostStruct{T}, ::Type{type}) where {T, type} = get(vr, type)
@inline Base.getindex(vr::CostStruct{T}, ::Type{type}) where {T, type} = get(vr, type)
@inline Base.push!(vr::CostStruct, v::T) where T = push!(get!(vr, T), v)
@inline Base.append!(vr::CostStruct, v::Vector{T}) where T = append!(get!(vr, T), v)

# Get the keys (tuple of types)
Base.keys(::CostStruct{T}) where T = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))

# Get a typed Tuple of vectors
Base.values(vr::CostStruct) = vr.data
valuetuple(vr::CostStruct, ::Type{T}) where T = vr.data

# Sum reduction
Base.sum(fun, vr::CostStruct; kw...) = sum(bindleadingargs(vrsum, fun, kw), values(vr); kw...)
vrsum(fun, kw, v::Vector) = sum(fun, v; kw...)

# Sum reduction over a subset of the elements
sumsubset(fun, vr::CostStruct, subsetfun; init=0.0) = sum(bindleadingargs(sumsubsetvec, fun, subsetfun, init), values(vr); init=init)
sumsubsetvec(fun, subsetfun, init, vector) = sum(fun, subsetfun(vector); init=init)
