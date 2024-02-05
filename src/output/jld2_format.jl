function saveU(::JLD2Format, U, filename::String)
    filename!="" && jldsave(filename; U=U.U)
    return nothing
end

function loadU!(::JLD2Format, U, filename::String)
    Unew = jldopen(filename, "r") do file
        file["U"]
    end
    @assert (length(Unew)==4 && size(Unew)==size(U)) "Size of supplied config is wrong"

    for ii in eachindex(U[1])
        for μ in 1:4
            U[μ,ii] = Unew[μ,ii]
        end
    end

    return nothing
end

function create_checkpoint(::JLD2Format, univ, updatemethod, filename::String)
    state = get_rng_state()
    filename!="" && jldsave(filename; univ=univ, updatemethod=updatemethod, rng=state)
    return nothing
end

function load_checkpoint(::JLD2Format, filename::String)
    univ, updatemethod, rng = jldopen(filename, "r") do file
        file["univ"], file["updatemethod"], file["state"]
    end
    Random.seed!(rng)
    return univ, updatemethod
end
