module MetaIO

using Dates
using JLD2
using LinearAlgebra
using OffsetArrays
using Polyester
using Random
using StaticArrays
using ..Logs
using ..Parameters
using ..Utils

import ..BiasModule: Bias, BiasSerialization, NoBias, calc_cv, recalc_cv!
import ..Fields: AbstractField, Gaugefield, Spinorfield, SpinorfieldEO, Paulifield
import ..Fields: Tensorfield, MultiSpinorfield, CPU, get_backend, bzeros, BACKENDS
import ..Fields: is_distributed, get_global_volume, get_global_dims, parallelfor
import ..Fields: WilsonGaugeAction, array_type, convert_field, device_to_host, allindices
import ..Universe: init_fermion_actions
import ..Updates: Updatemethod

export BMWFormat, BridgeFormat, Checkpointer, ConfigSaver, JLD2Format
export create_checkpoint, load_checkpoint, load_field!, save_field

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

function set_view!(fp, u, ::Type{T}; offset=0, infokws...) where {T}
    etype = Utils.MPI.Datatype(T)
    filetype = create_filetype(u, T)
    datarep = "native"
    Utils.MPI.File.set_view!(fp, offset, etype, filetype, datarep; infokws...)
    return nothing
end

function create_filetype(u, ::Type{T}) where {T}
    topology = u.topology
    inner_len = if u isa Spinorfield || u isa SpinorfieldEO || u isa Paulifield
        1
    elseif u isa Tensorfield
        6
    elseif u isa MultiSpinorfield
        u.numspinors
    else
        4
    end
    global_dims = (inner_len, topology.global_dims...)
    local_dims = (inner_len, topology.local_dims...)
    local_ranges = (1:inner_len, topology.bulk_sites.indices...) 
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

struct Checkpointer{T}
    checkpoint_dir::String
    checkpoint_every::Int64

    function Checkpointer(checkpoint_dir, checkpoint_every)
        if checkpoint_every > 0
            T = JLD2Format
            @level1("- Checkpoints will be created!")
            @level1("|  FORMAT: JLD2")
            @level1("|  DIRECTORY: $(checkpoint_dir)")
            @level1("|  INTERVAL: $(checkpoint_every)")
            @level1("-\n")
        else
            T = Nothing
        end

        return new{T}(checkpoint_dir, checkpoint_every)
    end
end

function create_checkpoint(
    cp::Checkpointer{T}, univ, updatemethod, updatemethod_pt, itrj; rank=mpi_myrank()
) where {T}
    T ≡ Nothing && return nothing
    instance = MPI_INSTANCE[]

    if itrj % cp.checkpoint_every == 0
        filename = joinpath(cp.checkpoint_dir, "checkpoint_$(instance)_$(rank).jld2")
        create_checkpoint(T(), univ, updatemethod, updatemethod_pt, itrj, filename)
        @level1("|")
        @level1("|  Checkpoint created in $(cp.checkpoint_dir)")
        @level1("|")
    end

    return nothing
end

function load_checkpoint(parameters; rank=mpi_myrank(), mpi_multi_sim=false, build=false)
    checkpoint_path = parameters.load_checkpoint_path
    @level1("[ Checkpoint loaded from $(checkpoint_path)\n")
    return load_checkpoint(JLD2Format(), parameters; rank, mpi_multi_sim, build)
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
            @level1("- Configs will be saved!")
            @level1("|  FORMAT: $(save_config_format)")
            @level1("|  DIRECTORY: $(save_config_dir)")
            @level1("|  INTERVAL: $(save_config_every)")
            @level1("-\n")
        end

        return new{T}(save_config_dir, save_config_every, ext)
    end
end

function save_field(saver::ConfigSaver{T}, U, itrj, parameters=nothing) where {T}
    T ≡ Nothing && return nothing

    if itrj % saver.save_config_every == 0
        itrjstring = lpad(itrj, 8, "0")
        filename = saver.save_config_dir * "/config_$(itrjstring)$(saver.ext)"
        save_field(T(), U, filename, parameters)
        @level1("|  Config saved in $(string(T)) in file \"$(filename)\"")
    end

    return nothing
end

function load_field!(U, parameters)
    parameters.load_config_fromfile || return false
    filename = parameters.load_config_path
    format = parameters.load_config_format

    try
        load_field!(FORMATS[parameters.load_config_format](), U, filename)
    catch _
        error("load_config_format \"$(format)\" not supported.")
    end

    @level1("[ Config loaded from $(filename)")
    return true
end

function get_rng_state()
    state = copy(Random.default_rng())
    return state
end

end
