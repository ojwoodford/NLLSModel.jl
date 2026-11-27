# A reusable setup of a model's linearization, for `optimize!(model, options; workspace)` (the SecondOrderOptimization
# extension). The extension builds the setup and the optimizer's workspace and defines the constructors; this file holds
# the type and the key that decides whether a stored setup is still the model's.
mutable struct ModelWorkspace{K, S, W}
    key::Union{Nothing, K}       # What the setup was built from (`setupkey`)
    setup::Union{Nothing, S}     # The objective, the start point and the functor that writes a solution back
    const optimizer::W           # The optimizer's own workspace (a SecondOrderOptimization.OptimizationWorkspace)
    ModelWorkspace{K, S, W}(key, setup, optimizer) where {K, S, W} = new{K, S, W}(key, setup, optimizer)
end

function Base.empty!(ws::ModelWorkspace)
    ws.key = nothing
    ws.setup = nothing
    empty!(ws.optimizer)
    return ws
end
Base.isempty(ws::ModelWorkspace) = isnothing(ws.setup)

# A hash of everything about a model's structure that its setup is built from: the number, types and sizes of the
# variables (a type decides `iseliminable`, so the partition), and for each cost its variables and, for a residual, its
# number of rows. Values are not hashed: the setup reads them from the model at each evaluation.
function structurehash(model::Model)
    h = hash(length(model.variables))
    for v in model.variables
        h = hash(dynamic(nvars(v)), hash(typeof(v), h))
    end
    return costshash(h, values(model.costs)...)
end
@inline costshash(h::UInt) = h
@inline costshash(h::UInt, costs::Vector, rest...) = costshash(costvechash(hash(length(costs), h), costs), rest...)
function costvechash(h::UInt, costs::Vector{C}) where C
    for c in costs
        for v in varindices(c)
            h = hash(v, h)
        end
        if C <: AbstractResidual
            h = hash(dynamic(nres(c)), h)
        end
    end
    return h
end

# The key of a setup: the model and its variable vector (the setup holds both), the structure's hash, and the keywords
# that the setup was built with. `unfixed` is copied when it is an array, so a change to the caller's array is seen.
setupkey(model::Model, keywords::NamedTuple) = (model, model.variables, structurehash(model), keycopy(keywords))
keycopy(keywords::NamedTuple) = merge(keywords, (unfixed = keywordcopy(keywords.unfixed),))
keywordcopy(x) = x
keywordcopy(x::AbstractArray) = copy(x)

# Whether a stored key is the key of `model` with `keywords`, computed without building that key (so without copying
# `unfixed`)
keymatches(::Nothing, model, keywords) = false
function keymatches(key::Tuple, model::Model, keywords::NamedTuple)
    key[1] === model || return false
    key[2] === model.variables || return false
    isequal(key[4], keywords) || return false
    return key[3] == structurehash(model)
end
