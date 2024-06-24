function saveU(::JLD2Format, U, filename::String)
    filename != "" && jldsave(filename; U=U.U)
    return nothing
end

function loadU!(::JLD2Format, U, filename::String)
    Unew = jldopen(filename, "r") do file
        file["U"]
    end
    @assert (size(Unew) == size(U.U)) "Size of supplied config is wrong"

    for ii in eachindex(U)
        for μ in 1:4
            U[μ, ii] = Unew[μ, ii]
        end
    end

    return nothing
end

function create_checkpoint(::JLD2Format, univ, updatemethods, itrj::Int, filename::String)
    state = get_rng_state()
    if filename != ""
        jldsave(filename; univ=univ, updatemethods=updatemethods, rngstate=state, itrj=itrj)
    end
    return nothing
end

function load_checkpoint(::JLD2Format, filename::String)
    univ, updatemethods, rngstate, itrj = jldopen(filename, "r") do file
        file["univ"], file["updatemethods"], file["rngstate"], file["itrj"]
    end
    copy!(Random.default_rng(), rngstate)
    
    if updatemethods isa Tuple
        updatemethod = updatemethods[1]
        updatemethod_pt = updatemethods[2]
    else
        updatemethod = updatemethods
        updatemethod_pt = nothing
    end

    return univ, updatemethod, updatemethod_pt, itrj
end
