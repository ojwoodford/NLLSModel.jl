
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

# Compile-time index of a type within a Tuple of types (returns 0 if not present)
@inline findtypeindex(::Type{target}, ::Type{target}, rest...) where target = 1
@inline function findtypeindex(::Type{target}, ::Type{other}, rest...) where {target, other}
    i = findtypeindex(target, rest...)
    return ifelse(i == 0, 0, i + 1)
end
@inline findtypeindex(::Type{target}) where target = 0
@inline typeindex(::Type{T}, ::Type{target}) where {T<:Tuple, target} = findtypeindex(target, fieldtypes(T)...)

struct CostStruct{T, V<:Tuple}
    data::V
    CostStruct{Union{}}() = new{Tuple{}, Tuple{}}(())
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

@inline function Base.get(cs::CostStruct{T}, ::Type{type})::Vector{type} where {T, type}
    idx = typeindex(T, type)
    idx > 0 || error("Type $type not in CostStruct{$T}")
    return @inbounds cs.data[idx]::Vector{type}
end

@inline Base.haskey(::CostStruct{T}, ::Type{type}) where {T, type} = typeindex(T, type) > 0
@inline Base.get!(cs::CostStruct{T}, ::Type{type}) where {T, type} = get(cs, type)
@inline Base.getindex(cs::CostStruct{T}, ::Type{type}) where {T, type} = get(cs, type)
@inline Base.push!(cs::CostStruct, v::T) where T = push!(get!(cs, T), v)
@inline Base.append!(cs::CostStruct, v::Vector{T}) where T = append!(get!(cs, T), v)

# Get the keys (tuple of types)
Base.keys(::CostStruct{T}) where T = ntuple(i -> fieldtype(T, i), Val(fieldcount(T)))

# Get a typed Tuple of vectors
Base.values(cs::CostStruct) = cs.data

# Sum reduction
struct _NoInit end
@inline _sum_tuple(fun::F, acc) where F = acc
@inline _sum_tuple(fun::F, acc, v::AbstractVector, rest...) where F = _sum_tuple(fun, isempty(v) ? acc : acc + sum(fun, v), rest...)
_sum_tuple_noinit(fun::F) where F = throw(ArgumentError("reducing over an empty CostStruct is not allowed; consider supplying `init`"))
@inline function _sum_tuple_noinit(fun::F, v::AbstractVector, rest...) where F
    isempty(v) && return _sum_tuple_noinit(fun, rest...)
    return _sum_tuple(fun, sum(fun, v), rest...)
end
Base.sum(fun::F, cs::CostStruct; init=_NoInit()) where F = init isa _NoInit ? _sum_tuple_noinit(fun, values(cs)...) : _sum_tuple(fun, init, values(cs)...)
Base.sum(cs::CostStruct; kw...) = sum(identity, cs; kw...)

# Sum reduction over a subset of the elements
@inline _sumsubset_tuple(fun::F, subsetfun::S, acc) where {F, S} = acc
@inline function _sumsubset_tuple(fun::F, subsetfun::S, acc, v::AbstractVector, rest...) where {F, S}
    sub = subsetfun(v)
    return _sumsubset_tuple(fun, subsetfun, isempty(sub) ? acc : acc + sum(fun, sub), rest...)
end
sumsubset(fun::F, cs::CostStruct, subsetfun::S; init=0.0) where {F, S} = _sumsubset_tuple(fun, subsetfun, init, values(cs)...)
