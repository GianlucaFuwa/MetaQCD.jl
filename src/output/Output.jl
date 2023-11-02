module Output

using JLD2
using LinearAlgebra
using Random
using StaticArrays

export VerboseLevel, Verbose1, Verbose2, Verbose3
export print_verbose1, print_verbose2, print_verbose3,
    println_verbose1, println_verbose2, println_verbose3
export SaveConfigs, loadU_bridge!, loadU_jld!, load_gaugefield!, save_gaugefield,
    saveU_bridge, saveU_jld

include("verbose.jl")

# struct BMWFormat end
struct BridgeFormat end
struct JLD2Format end

# include("bmw_format.jl")
include("bridge_format.jl")
include("jld2_format.jl")

struct SaveConfigs{T}
    saveU_format::Union{Nothing, String}
    saveU_dir::String
    saveU_every::Int64
    itrjsavecount::Base.RefValue{Int64}

    function SaveConfigs(saveU_format, saveU_dir, saveU_every, vp)
        itrjsavecount = Base.RefValue{Int64}(0)

        if saveU_format == "bridge"
            T = BridgeFormat
        elseif saveU_format == "jld" || saveU_format == "jld2"
            T = JLD2Format
        elseif saveU_format === nothing
            T = Nothing
        else
            error("saveU format $saveU_format not supported")
        end

        if T !== Nothing
            println_verbose1(vp, "\t>> Save config every $(saveU_every) trajectory!")
        end

        return new{T}(saveU_format, saveU_dir, saveU_every, itrjsavecount)
    end
end

function save_gaugefield(save_configs::SaveConfigs{T}, U, verbose, itrj) where {T}
    T===Nothing && return nothing

    if itrj % save_configs.saveU_every == 0
        save_configs.itrjsavecount[] += 1
        itrjstring = lpad(itrj, 8, "0")

        if T == JLD2Format
            filename = save_configs.saveU_dir * "/config_$(itrjstring).jld2"
            saveU_jld(U, filename)
        elseif T == BridgeFormat
            filename = save_configs.saveU_dir * "/config_$(itrjstring).txt"
            saveU_bridge(U, filename)
        end
        rng_seed = get_current_seed()
        println_verbose1(verbose, ">> Stored config. Current rng: $(string.(rng_seed))")
    end

    return nothing
end

function load_gaugefield!(U, parameters, verbose)
    parameters.loadU_fromfile || return false
    filename = parameters.loadU_dir * "/" * parameters.loadU_filename
    format = parameters.loadU_format

    if format == "bridge"
        loadU_bridge!(U, filename)
    elseif format ∈ ("jld", "jld2")
        loadU_jld!(U, filename)
    else
        error("loadU_format \"$(format)\" not supported.")
    end

    println_verbose1(verbose, ">> Config loaded from $(filename)")
    return true
end

function get_current_seed()
    fn = fieldnames(Xoshiro)
    seed = [getfield(copy(Random.default_rng()), name) for name in fn]
    return seed
end

end
