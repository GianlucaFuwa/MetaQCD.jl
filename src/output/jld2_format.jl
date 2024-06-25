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

function create_checkpoint(
    ::JLD2Format, univ, updatemethod, updatemethod_pt, itrj::Int, filename::String
)
    state = get_rng_state()
    if filename != ""
        jldsave(
            filename; 
            U=univ.U,
            fermion_actions=univ.fermion_actions,
            bias=univ.bias,
            numinstances=univ.numinstances,
            updatemethod=updatemethod,
            updatemethod_pt=updatemethod_pt,
            rngstate=state,
            itrj=itrj,
        )
    end
    return nothing
end

function load_checkpoint(::JLD2Format, filename::String)
    U, fermion_actions, bias, numinstances, updatemethod, updatemethod_pt, rngstate, itrj =
        jldopen(filename, "r") do file
        file["U"], file["fermion_actions"], file["bias"], file["numinstances"],
        file["updatemethod"], file["updatemethod_pt"], file["rngstate"], file["itrj"]
    end

    copy!(Random.default_rng(), rngstate)
    return U, fermion_actions, bias, numinstances, updatemethod, updatemethod_pt, itrj
end
