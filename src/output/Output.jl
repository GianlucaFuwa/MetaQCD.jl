module Output

using Dates
using Format
using InteractiveUtils: InteractiveUtils
using JLD2
using KernelAbstractions # TODO: save and load of GPUD
using MPI
using LinearAlgebra
using Printf
using Random
using StaticArrays
using ..Utils: restore_last_row

export __GlobalLogger, MetaLogger, current_time, @level1, @level2, @level3
export BMWFormat, BridgeFormat, Checkpointer, ConfigSaver, JLD2Format, set_global_logger!
export fclose, fopen, printf, prints_to_console
export create_checkpoint, load_checkpoint, load_config!, save_config

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const COMM_SIZE = MPI.Comm_size(COMM)
const MYRANK = MPI.Comm_rank(COMM)

include("cio.jl")
include("verbose.jl")

abstract type AbstractFormat end
struct BMWFormat <: AbstractFormat end
struct BridgeFormat <: AbstractFormat end
struct JLD2Format <: AbstractFormat end

const FORMATS = Dict{String, Any}(
    "bmw" => BMWFormat,
    "bridge" => BridgeFormat,
    "jld" => JLD2Format,
    "jld2" => JLD2Format,
    "" => Nothing,
)

const EXT = Dict{String, String}(
    "bmw" => ".bmw",
    "bridge" => ".txt",
    "jld" => ".jld2",
    "jld2" => ".jld2",
    "" => "",
)

include("bmw_format.jl")
include("bridge_format.jl")
include("jld2_format.jl")

@inline current_time() = Dates.now(UTC)

struct Checkpointer{T}
    checkpoint_dir::String
    checkpoint_every::Int64

    function Checkpointer(checkpoint_dir, checkpoint_every)
        if checkpoint_every > 0
            T = JLD2Format
            @level1("┌ Checkpoints will be created!")
            @level1("|  FORMAT: JLD2")
            @level1("|  DIRECTORY: $(checkpoint_dir)")
            @level1("|  INTERVAL: $(checkpoint_every)")
            @level1("└\n")
        else
            T = Nothing
        end

        return new{T}(checkpoint_dir, checkpoint_every)
    end
end

function create_checkpoint(
    cp::Checkpointer{T}, univ, updatemethod, updatemethod_pt, itrj
) where {T}
    T ≡ Nothing && return nothing

    if itrj % cp.checkpoint_every == 0
        filename = cp.checkpoint_dir * "/checkpoint_$(MYRANK).jld2"
        create_checkpoint(T(), univ, updatemethod, updatemethod_pt, itrj, filename)
        @level1("|")
        @level1("|  Checkpoint created in $(filename)")
        @level1("|")
    end

    return nothing
end

function load_checkpoint(checkpoint_path)
    @level1("[ Checkpoint loaded from $(checkpoint_path)\n")
    if COMM_SIZE > 1
        checkpoint_file = checkpoint_path * "_$(MYRANK).jld2"
    else
        checkpoint_file = checkpoint_path * ".jld2"
    end
    return load_checkpoint(JLD2Format(), checkpoint_file)
end

struct ConfigSaver{T}
    save_config_dir::String
    save_config_every::Int64
    ext::String
    function ConfigSaver(save_config_format, save_config_dir, save_config_every)
        T, ext = try
            FORMATS[save_config_format], EXT[save_config_format]
        catch _
            error("save_config_format $save_config_format not supported")
        end

        if save_config_every <= 0
            T = Nothing
        end

        if T ≢ Nothing
            @level1("┌ Configs will be saved!")
            @level1("|  FORMAT: $(save_config_format)")
            @level1("|  DIRECTORY: $(save_config_dir)")
            @level1("|  INTERVAL: $(save_config_every)")
            @level1("└\n")
        end

        return new{T}(save_config_dir, save_config_every, ext)
    end
end

function save_config(saver::ConfigSaver{T}, U, itrj) where {T}
    T ≡ Nothing && return nothing

    if itrj % saver.save_config_every == 0
        itrjstring = lpad(itrj, 8, "0")
        filename = saver.save_config_dir * "/config_$(itrjstring)$(saver.ext)"
        save_config(T(), U, filename)
        @level1("|  Config saved in $T in file \"$(filename)\"")
    end

    return nothing
end

function load_config!(U, parameters)
    parameters.load_config_fromfile || return false
    filename = parameters.loadU_dir * "/" * parameters.load_config_filename
    format = parameters.load_config_format

    try
        load_config!(FORMATS[parameters.load_config_format](), U, filename)
    catch _
        error("loadU_format \"$(format)\" not supported.")
    end

    @level1("[ Config loaded from $(filename)")
    return true
end

function get_rng_state()
    state = copy(Random.default_rng())
    return state
end

end
