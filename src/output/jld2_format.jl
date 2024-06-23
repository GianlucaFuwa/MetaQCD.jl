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

function create_checkpoint(::JLD2Format, univ, updatemethods, filename::String)
    rng = get_rng_state()
    filename != "" && jldsave(filename; univ=univ, updatemethods=updatemethods, rng=rng)
    return nothing
end

function load_checkpoint(::JLD2Format, filename::String)
    univ, updatemethods, rng = jldopen(filename, "r") do file
        file["univ"], file["updatemethods"], file["rng"]
    end
    Random.seed!(rng)
    
    if updatemethods isa Tuple
        updatemethod = updatemethods[1]
        updatemethod_pt = updatemethods[2]
    end
    return univ, updatemethod, updatemethod_pt
end
