function saveU(::JLD2Format, U, filename)
    save(filename, "U", U.U)
    return nothing
end

function loadU!(::JLD2Format, U, filename)
    Unew = load(filename, "U")
    @assert (length(Unew)==4 && size(Unew)==size(U)) "Size of supplied config is wrong"

    for ii in eachindex(U[1])
        for μ in 1:4
            U[μ][ii] = Unew[μ][ii]
        end
    end

    return nothing
end

function create_checkpoint(::JLD2Format, univ, updatemethod, filename)
    state = get_rng_state()
    save(filename, "univ", univ, "updatemethod", updatemethod, "rng", state)
    return nothing
end

function load_checkpoint(::JLD2Format, filename)
    univ, updatemethod, rng = load(filename, "univ", "updatemethod", "rng")
    Random.seed!(rng)
    return univ, updatemethod
end
