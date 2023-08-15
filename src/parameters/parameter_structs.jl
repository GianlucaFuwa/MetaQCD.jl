import ..Measurements: MeasurementParameters
import ..Measurements: construct_measurement_parameters_from_dict
import ..Measurements: PlaquetteParameters, PolyakovParameters
import ..Measurements: prepare_measurement_from_dict
import ..Measurements: TopologicalChargeParameters, WilsonLoopParameters

const important_parameters = [
    "L",
    "β",
    "kind_of_gaction",
    "update_method",
    "meta_enabled",
    "kind_of_cv",
    "numCVsmear",
    "metro_ϵ",
    "hmc_steps",
    "hmc_Δτ",
    "tempering_enabled",
    "methodname",
    "measurement_basedir",
    "has_gradient_flow",
    "measurement_dir",
    "kinds_of_topological_charge",
    "measurements_for_flow",
    "flow_measurements",
]

function check_important_parameters(key)
    findornot = findfirst(x -> x==key, important_parameters)

    if findornot === nothing
        return false
    else
        return true
    end

    return nothing
end

function struct2dict(x::T) where {T}
    dict = Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))
    return dict
end

function generate_printlist(x::Type)
    pnames = fieldnames(x)
    plist = String[]

    for i in eachindex(pnames)
        push!(plist, String(pnames[i]))
    end

    return plist
end

Base.@kwdef mutable struct PrintPhysicalParameters
    L::NTuple{4, Int64} = (4, 4, 4, 4)
    beta::Float64 = 5.7
    NC::Int64 = 3
    kind_of_gaction::String = "wilson"
    numtherm::Int64 = 10
    numsteps::Int64 = 100
    inital::String = "cold"
    update_method::Vector{String} = ["HMC"]
    hb_maxit::Int64 = 10^5
    numheatbath::Int64 = 4
    metro_epsilon::Float64 = 0.1
    metro_multi_hit::Int64 = 1
    metro_target_acc::Float64 = 0.5
    eo::Bool = false
    numorelax::Int64 = 0
    parity_update::Bool = false
end

Base.@kwdef mutable struct PrintMetaParameters
    meta_enabled::Bool = false
    tempering_enabled::Bool = false
    numinstances::Int64 = 1
    swap_every::Int64 = 1
    non_metadynamics_updates::Int64 = 1
    measure_on_all::Bool = false
    kind_of_cv::String = "clover"
    numsmears_for_cv::Int64 = 4
    rhostout_for_cv::Float64 = 0.125
    symmetric::Bool = false
    cvlims::NTuple{2, Float64} = (-7, 7)
    bin_width::Float64 = 1e-2
    meta_weight::Float64 = 1e-3
    penalty_weight::Float64 = 1000.0
    wt_factor::Float64 = Inf
    is_static::Union{Bool, Vector{Bool}} = false
    kinds_of_weights::Vector{String} = ["tiwari"]
    usebiases::Union{Nothing, String, Vector{Union{Nothing,String}}} = nothing
end

Base.@kwdef mutable struct PrintSystemParameters
    log_dir::String = ""
    logfile::String = ""
    verboselevel::Int64 = 1
    loadU_format::Union{Nothing, String} = nothing
    loadU_dir::String = ""
    loadU_fromfile::Bool = false
    loadU_filename::String = ""
    saveU_dir::String = ""
    saveU_format::Union{String, Nothing} = nothing
    saveU_every::Int64 = 1
    randomseed::Int64 = 0
    measurement_basedir::String = ""
    measurement_dir::String = ""
    bias_basedir::Union{Nothing, String, Vector{String}} = nothing
    bias_dir::Union{Nothing, String, Vector{Union{Nothing,String}}} = nothing
    overwrite::Bool = false
end

Base.@kwdef mutable struct PrintHMCParameters
    hmc_deltatau::Float64 = 0.1
    hmc_steps::Int64 = 10
    hmc_integrator::String = "Leapfrog"
    hmc_numsmear::Int64 = 0
    hmc_rhostout::Float64 = 0.0
end

Base.@kwdef mutable struct PrintGradientFlowParameters
    hasgradientflow::Bool = false
    flow_integrator::String = "euler"
    flow_num::Int64 = 1
    flow_tf::Float64 = 0.1
    flow_steps::Int64 = 10
    flow_measure_every::Int64 = 1
end

Base.@kwdef mutable struct PrintMeasurementParameters
    measurement_method::Vector{Dict} = Dict[]
end

const printlist_physical = generate_printlist(PrintPhysicalParameters)
const printlist_meta = generate_printlist(PrintMetaParameters)
const printlist_system = generate_printlist(PrintSystemParameters)
const printlist_hmc = generate_printlist(PrintHMCParameters)
const printlist_measurement = generate_printlist(PrintMeasurementParameters)

abstract type SmearingParameters end

Base.@kwdef mutable struct NoSmearingParameters <: SmearingParameters end

Base.@kwdef mutable struct MeasurementParameterSet
    measurement_methods::Vector{MeasurementParameters} = []
end

function transform_measurement_dictvec(value)
    flow_dict = Dict()
    nummeasure = length(value)
    value_out = Vector{Measurement_parameters}(undef, nummeasure)
    hasgradientflow = false

    for i in 1:nummeasure
        if haskey(value[i], "methodname")
            if value[i]["methodname"] == "Topological_charge"
                hasgradientflow = true
                value_out[i] = transform_topological_charge_measurement!(
                    flow_dict,
                    value[i],
                )
            else
                value_out[i] = construct_Measurement_parameters_from_dict(value[i])
            end
        else
            error("method name in measurement should be set")
        end
    end

    return value_out, smear_dict, hasgradientflow
end

function transform_topological_charge_measurement!(flow_dict, measurement)
    @assert haskey(measurement, "methodname") "method name in measurement should be set"
    @assert measurement["methodname"] == "Topological_charge" "function is for top. charge"

    measurement_revised = Dict()

    for (key, value) in measurement
        if key == "flow_integrator"
            flow_dict["flow_integrator"] = value
        elseif key == "flow_num"
            flow_dict["flow_num"] = value
        elseif key == "flow_tf"
            flow_dict["flow_tf"] = value
        elseif key == "flow_steps"
            flow_dict["flow_steps"] = value
        elseif key == "flow_measure_every"
            flow_dict["flow_measure_every"] = value
        else
            measurement_revised[key] = value
        end
    end

    value_m = construct_Measurement_parameters_from_dict(measurement_revised)
    flow_dict["measurements_for_flow"] = Dict()
    flow_dict["measurements_for_flow"]["Topological_charge"] = measurement_revised

    return value_m
end

function construct_printable_parameters_fromdict!(
    key,
    value,
    physical,
    meta,
    system,
    hmc,
)
    if key == "L"
        value = collect(value)
    elseif key == "r"
        value = Float64(value)
    end

    hasvalue = false
    pname_i = Symbol(key)
    physical_index = findfirst(x -> x==key, printlist_physical)

    if physical_index !== nothing
        setfield!(physical, pname_i, value)
        hasvalue = true
    end

    meta_index = findfirst(x -> x==key, printlist_meta)

    if meta_index !== nothing
        setfield!(meta, pname_i, value)
        hasvalue = true
    end

    system_index = findfirst(x -> x==key, printlist_system)

    if system_index !== nothing
        setfield!(system, pname_i, value)
        hasvalue = true
    end

    hmc_index = findfirst(x -> x==key, printlist_hmc)

    if hmc_index !== nothing
        setfield!(hmc, pname_i, value)
        hasvalue = true
    end

    if hasvalue == false
        @warn "$(key) is not used"
    end

    return hasvalue
end

function construct_printable_parameters_fromdict!(
    x::Dict,
    physical,
    meta,
    system,
    hmc
)
    for (key, value) in x
        hasvalue = false
        pname_i = Symbol(key)
        physical_index = findfirst(x -> x==pname_i, printlist_physical)

        if physical_index !== nothing
            setfield!(physical, pname_i, value)
            hasvalue = true
        end

        meta_index = findfirst(x -> x==pname_i, printlist_meta)

        if meta_index !== nothing
            setfield!(meta, pname_i, value)
            hasvalue = true
        end

        system_index = findfirst(x -> x==pname_i, printlist_system)

        if system_index !== nothing
            setfield!(system, pname_i, value)
            hasvalue = true
        end

        hmc_index = findfirst(x -> x==pname_i, printlist_hmc)

        if hmc_index !== nothing
            setfield!(hmc, pname_i, value)
            hasvalue = true
        end

        if hasvalue == false
            @warn "$(pname_i) is not used"
        end
    end

    return nothing
end

function remove_default_values!(x::Dict, defaultsystem)
    for (key, value) in x
        if hasfield(typeof(defaultsystem), Symbol(key))
            default_value = getfield(defaultsystem,Symbol(key))

            if value == default_value || string(value) == string(default_value)
                if check_important_parameters(key) == false
                    delete!(x, key)
                end
            else
                if value === nothing
                    x[key] = "nothing"
                end
            end

        else
            if value === nothing
                x[key] = "nothing"
            end
        end

        if typeof(value) == Vector{Measurement_parameters}
            construct_dict_from_measurement!(x, value)
        end
    end

    return nothing
end

function construct_dict_from_measurement!(x, value)
    measuredic = Dict()

    for measure in value
        methoddic = struct2dict(measure)
        measure_struct_default = typeof(measure)()
        remove_default_values!(methoddic, measure_struct_default)
        measuredic[methoddic["methodname"]] = methoddic
    end

    x["measurement_methods"] = measuredic
    return nothing
end

function remove_default_values!(x::Dict)
    physical = Print_physical_parameters()
    meta = Print_meta_parameters()
    system = Print_system_parameters()
    hmc = Print_hmc_parameters()

    for (params, paramsname) in x
        remove_default_values!(x[params], physical)
        remove_default_values!(x[params], meta)
        remove_default_values!(x[params], system)
        remove_default_values!(x[params], hmc)
    end

    return nothing
end
