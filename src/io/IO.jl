module MetaIO

using Dates
using InteractiveUtils: InteractiveUtils
using JLD2
using KernelAbstractions # TODO: save and load of Fields on GPUs
using LinearAlgebra
using Polyester
using Printf
using Random
using StaticArrays
using ..Parameters
using ..Utils

export __GlobalLogger, MetaLogger, current_time, @level1, @level2, @level3
export BMWFormat, BridgeFormat, Checkpointer, ConfigSaver, JLD2Format, set_global_logger!
export fclose, fopen, printf, prints_to_console, newline
export create_checkpoint, load_checkpoint, load_config!, save_config

include("printf.jl")
include("verbose.jl")

abstract type AbstractFormat end
struct BMWFormat <: AbstractFormat end
struct BridgeFormat <: AbstractFormat end
# TODO: struct ILDGFormat <: AbstractFormat end
struct JLD2Format <: AbstractFormat end
struct MPIFormat <: AbstractFormat end

const FORMATS = Dict{String, Any}(
    "bmw" => BMWFormat,
    "bridge" => BridgeFormat,
    # "ildg" => ILDGFormat,
    "jld" => JLD2Format,
    "jld2" => JLD2Format,
    "mpi" => MPIFormat,
    "" => Nothing,
)

const EXT = Dict{String, String}(
    "bmw" => ".bmw",
    "bridge" => ".txt",
    # "ildg" => ".ildg",
    "jld" => ".jld2",
    "jld2" => ".jld2",
    "mpi" => ".bin",
    "" => "",
)

function proc_offset(args...) end # INFO: Need this for writing fields to file --- is implemented in fields/parallel.jl

function set_view!(fp, U, ::Type{T}; offset=0, infokws...) where {T}
    etype = Utils.MPI.Datatype(T)
    filetype = create_filetype(U, T)
    datarep = "native"
    Utils.MPI.File.set_view!(fp, offset, etype, filetype, datarep; infokws...)
    return nothing
end

function create_filetype(U, ::Type{T}) where {T}
    topology = U.topology
    # 18 entries in matrix * 4 directions per site
    global_dims = (4, topology.global_dims...)
    local_dims = (4, topology.local_dims...)
    local_ranges = (1:4, topology.local_ranges...) 
    offsets = map(r -> (first(r) - 1), local_ranges)
    oldtype = Utils.MPI.Datatype(T)
    ftype = Utils.MPI.Types.create_subarray(global_dims, local_dims, offsets, oldtype)
    Utils.MPI.Types.commit!(ftype)
    return ftype
end

include("bmw_format.jl")
include("bridge_format.jl")
include("jld2_format.jl")
include("mpi_format.jl")

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
        filename = cp.checkpoint_dir * "/checkpoint_$(mpi_myrank()).jld2"
        create_checkpoint(T(), univ, updatemethod, updatemethod_pt, itrj, filename)
        @level1("|")
        @level1("|  Checkpoint created in $(filename)")
        @level1("|")
    end

    return nothing
end

function load_checkpoint(checkpoint_path)
    @level1("[ Checkpoint loaded from $(checkpoint_path)\n")
    if mpi_size() > 1
        checkpoint_file = checkpoint_path * "_$(mpi_myrank()).jld2"
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
