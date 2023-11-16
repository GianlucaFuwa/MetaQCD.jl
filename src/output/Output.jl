module Output

using JLD2
using LinearAlgebra
using Random
using StaticArrays

export VerboseLevel, Verbose1, Verbose2, Verbose3
export print_verbose1, print_verbose2, print_verbose3,
    println_verbose1, println_verbose2, println_verbose3
export BridgeFormat, Checkpointer, JLD2Format, SaveConfigs
export load_checkpoint, load_gaugefield!, loadU!, save_gaugefield, saveU

include("verbose.jl")

# struct BMWFormat end
struct BridgeFormat end
struct JLD2Format end

# include("bmw_format.jl")
include("bridge_format.jl")
include("jld2_format.jl")

struct Checkpointer{T}
    checkpoint_dir::String
    checkpoint_every::Int64
    ext::String

    function Checkpointer(checkpointing_enabled, checkpoint_dir, checkpoint_every, vp)

        if checkpointing_enabled
            T == JLD2Format
            ext = ".jld2"
        else
            T == Nothing
            ext = ""
        end

        if T ≢ Nothing
            println_verbose1(vp, "\t>> Checkpointing every $(checkpoint_every) trajectory!")
        end

        return new{T}(checkpoint_dir, checkpoint_every, ext)
    end
end

function (cp::Checkpointer{T})(univ, updatemethod, verbose, itrj) where {T}
    T≡Nothing && return nothing

    if itrj%cp.checkpoint_every == 0
        filename = cp.checkpoint_dir * "/checkpoint$(cp.ext)"
        create_checkpoint(T(), univ, updatemethod, filename)
        println_verbose1(verbose, ">> Checkpoint created in $(filename)")
    end

    return nothing
end

function load_checkpoint(filename)
    return load_checkpoint(JLD2Format(), filename)
end

struct SaveConfigs{T}
    saveU_dir::String
    saveU_every::Int64
    ext::String

    function SaveConfigs(saveU_format, saveU_dir, saveU_every, vp)

        if saveU_format == "bridge"
            T = BridgeFormat
            ext = ".txt"
        elseif saveU_format == "jld" || saveU_format == "jld2"
            T = JLD2Format
            ext = ".jld2"
        elseif saveU_format ≡ nothing
            T = Nothing
            ext = ""
        else
            error("saveU format $saveU_format not supported")
        end

        if T ≢ Nothing
            println_verbose1(vp, "\t>> Save config every $(saveU_every) trajectory!")
        end

        return new{T}(saveU_dir, saveU_every, ext)
    end
end

function save_gaugefield(saver::SaveConfigs{T}, U, verbose, itrj) where {T}
    T≡Nothing && return nothing

    if itrj%saver.saveU_every == 0
        itrjstring = lpad(itrj, 8, "0")
        filename = saver.saveU_dir * "/config_$(itrjstring)$(saver.ext)"
        saveU(T(), U, filename)
        println_verbose1(verbose, ">> Config saved in $(filename)")
    end

    return nothing
end

function load_gaugefield!(U, parameters, verbose)
    parameters.loadU_fromfile || return false
    filename = parameters.loadU_dir * "/" * parameters.loadU_filename
    format = parameters.loadU_format

    if format == "bridge"
        loadU!(BridgeFormat(), U, filename)
    elseif format ∈ ("jld", "jld2")
        loadU!(JLD2Format(), U, filename)
    else
        error("loadU_format \"$(format)\" not supported.")
    end

    println_verbose1(verbose, ">> Config loaded from $(filename)")
    return true
end

function get_rng_state()
    rng = copy(Random.default_rng())
    state = [getfield(rng, i) for i in 1:4]
    return state
end

end
