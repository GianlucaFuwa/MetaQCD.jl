module Output

using Dates
using JLD2
using KernelAbstractions # TODO: save and load of GPUD
using LinearAlgebra
using Random
using StaticArrays
using ..Utils: restore_last_row

export MetaLogger, current_time, @level1, @level2, @level3, set_global_logger!
export BMWFormat, BridgeFormat, Checkpointer, JLD2Format, SaveConfigs
export load_checkpoint, load_gaugefield!, loadU!, save_gaugefield, saveU

include("verbose.jl")

abstract type AbstractFormat end
struct BMWFormat <: AbstractFormat end
struct BridgeFormat <: AbstractFormat end
struct JLD2Format <: AbstractFormat end

include("bmw_format.jl")
include("bridge_format.jl")
include("jld2_format.jl")

const date_format = "yyyy-mm-dd HH:MM:SS"

current_time() = "$(Dates.format(now(), date_format))"

struct Checkpointer{T}
    checkpoint_dir::String
    checkpoint_every::Int64
    ext::String

    function Checkpointer(checkpointing_enabled, checkpoint_dir, checkpoint_every)
        if checkpointing_enabled
            T = JLD2Format
            ext = ".jld2"
            @level1("[ Checkpointing every $(checkpoint_every) trajectory!")
        else
            T = Nothing
            ext = ""
        end

        return new{T}(checkpoint_dir, checkpoint_every, ext)
    end
end

function (cp::Checkpointer{T})(univ, updatemethod, itrj) where {T}
    T ≡ Nothing && return nothing

    if itrj % cp.checkpoint_every == 0
        filename = cp.checkpoint_dir * "/checkpoint$(cp.ext)"
        create_checkpoint(T(), univ, updatemethod, filename)
        @level1("|  Checkpoint created in $(filename)")
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
    function SaveConfigs(saveU_format, saveU_dir, saveU_every)
        if saveU_format == "bridge"
            T = BridgeFormat
            ext = ".txt"
        elseif saveU_format == "jld" || saveU_format == "jld2"
            T = JLD2Format
            ext = ".jld2"
        elseif saveU_format == "bmw"
            T = BMWFormat
            ext = ".bmw"
        elseif saveU_format == ""
            T = Nothing
            ext = ""
        else
            error("saveU format $saveU_format not supported")
        end

        if T ≢ Nothing
            @level1("[ Save config every $(saveU_every) trajectory!")
        end

        return new{T}(saveU_dir, saveU_every, ext)
    end
end

function save_gaugefield(saver::SaveConfigs{T}, U, itrj) where {T}
    T ≡ Nothing && return nothing

    if itrj % saver.saveU_every == 0
        itrjstring = lpad(itrj, 8, "0")
        filename = saver.saveU_dir * "/config_$(itrjstring)$(saver.ext)"
        saveU(T(), U, filename)
        @level1("|  Config saved in $T in file \"$(filename)\"")
    end

    return nothing
end

function load_gaugefield!(U, parameters)
    parameters.loadU_fromfile || return false
    filename = parameters.loadU_dir * "/" * parameters.loadU_filename
    format = parameters.loadU_format

    if format == "bridge"
        loadU!(BridgeFormat(), U, filename)
    elseif format ∈ ("jld", "jld2")
        loadU!(JLD2Format(), U, filename)
    elseif format == "bmw"
        loadU!(BMWFormat(), U, filename)
    else
        error("loadU_format \"$(format)\" not supported.")
    end

    @level1("[ Config loaded from $(filename)")
    return true
end

function get_rng_state()
    rng = copy(Random.default_rng())
    state = [getfield(rng, i) for i in 1:4]
    return state
end

end
