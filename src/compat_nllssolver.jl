# NLLSsolver.jl compatibility. Remove (or move to NLLSsolver.jl) when NLLSsolver is retired.
#
# Inside `optimize!` callbacks, `problem.varnext` and `problem.varbest` expose the trial / best variables that
# NLLSsolver keeps in task-local storage. Outside such a callback they return `problem.variables`.
function Base.getproperty(p::Model, s::Symbol)
    if s === :varnext
        if haskey(task_local_storage(), :nlls_varnext)
            val = task_local_storage(:nlls_varnext)
            if val isa Vector
                return val
            else
                v = copy(getfield(p, :variables))
                unfixed = get(task_local_storage(), :nlls_unfixed, 1)
                v[unfixed] = val
                return v
            end
        end
        return getfield(p, :variables)
    elseif s === :varbest
        if haskey(task_local_storage(), :nlls_varbest)
            val = task_local_storage(:nlls_varbest)
            if val isa Vector
                return val
            else
                v = copy(getfield(p, :variables))
                unfixed = get(task_local_storage(), :nlls_unfixed, 1)
                v[unfixed] = val
                return v
            end
        end
        return getfield(p, :variables)
    end
    return getfield(p, s)
end

