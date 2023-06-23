struct2dict(x::T) where {T} =
    Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))

abstract type MeasurementParameters end

Base.@kwdef mutable struct GaugeActionParameters <: MeasurementParameters
    methodname::String = "gauge_action"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
    kinds_of_gaction::Vector{String} = ["wilson"]
end

Base.@kwdef mutable struct PlaquetteParameters <: MeasurementParameters
    methodname::String = "plaquette"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
end

Base.@kwdef mutable struct PolyakovParameters <: MeasurementParameters
    methodname::String = "polyakov_loop"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
end

Base.@kwdef mutable struct WilsonLoopParameters <: MeasurementParameters
    methodname::String = "wilson_loop"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    Tmax::Int64 = 4
    Rmax::Int64 = 4
    measure_every::Int64 = 10
end

Base.@kwdef mutable struct TopologicalChargeParameters <: MeasurementParameters
    methodname::String = "topological_charge"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 10
    kinds_of_topological_charge::Vector{String} = ["clover"]
end

Base.@kwdef mutable struct EnergyDensityParameters <: MeasurementParameters
    methodname::String = "energy_density"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 10
    kinds_of_energy_density::Vector{String} = ["clover"]
end

Base.@kwdef mutable struct MetaChargeParameters <: MeasurementParameters
    methodname::String = "meta_charge"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
end

Base.@kwdef mutable struct MetaWeightParameters <: MeasurementParameters
    methodname::String = "meta_weight"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
    weight_method::Vector{String} = ["kish"]
end

function initialize_measurement_parameters(methodname)
    if Unicode.normalize(methodname, casefold = true) == "gauge_action"
        method = GaugeActionParameters()
    elseif Unicode.normalize(methodname, casefold = true) == "plaquette"
        method = PlaquetteParameters()
    elseif Unicode.normalize(methodname, casefold = true) == "polyakov_loop"
        method = PolyakovParameters()
    elseif Unicode.normalize(methodname, casefold = true) == "wilson_loop"
        method = WilsonLoopParameters()
    elseif Unicode.normalize(methodname, casefold = true) == "topological_charge"
        method = TopologicalChargeParameters()
    elseif Unicode.normalize(methodname, casefold = true) == "energy_density"
        method = EnergyDensityParameters()
    elseif Unicode.normalize(methodname, casefold = true) == "meta_charge"
        method = MetaChargeParameters()
    else
        error("$methodname is not implemented")
    end
    return method
end

function prepare_measurement_from_dict(U, value_i::Dict, filename = "", flow = false)
    parameters = construct_measurement_parameters_from_dict(value_i)
    return prepare_measurement(U, parameters, filename, flow)
end

function construct_measurement_parameters_from_dict(value_i::Dict)
    @assert haskey(value_i, "methodname") "methodname should be in measurement."
    methodname = value_i["methodname"]
    method = initialize_measurement_parameters(methodname)
    method_dict = struct2dict(method)

    for (key_ii, value_ii) in value_i
        if haskey(method_dict, key_ii)
            if typeof(value_ii) !== nothing
                keytype = typeof(getfield(method, Symbol(key_ii)))
                setfield!(method, Symbol(key_ii), keytype(value_ii))
            end
        end
    end

    value_out = deepcopy(method)
    return value_out
end

function prepare_measurement(
    U,
    meas_parameters::T,
    filename = "",
    flow = false,
) where {T}
    if T == GaugeActionParameters
        filename_input = ifelse(filename == "", "gauge_action.txt", filename)
        measurement = GaugeActionMeasurement(U, meas_parameters, filename_input, flow)
    elseif T == PlaquetteParameters
        filename_input = ifelse(filename == "", "plaquette.txt", filename)
        measurement = PlaquetteMeasurement(U, meas_parameters, filename_input, flow)
    elseif T == PolyakovParameters
        filename_input = ifelse(filename == "", "polyakov_loop.txt", filename)
        measurement = PolyakovMeasurement(U, meas_parameters, filename_input, flow)
    elseif T == TopologicalChargeParameters
        filename_input = ifelse(filename == "", "topological_charge.txt", filename)
        measurement = TopologicalChargeMeasurement(U, meas_parameters, filename_input, flow)
    elseif T == WilsonLoopParameters
        filename_input = ifelse(filename == "", "wilson_loop.txt", filename)
        measurement = WilsonLoopMeasurement(U, meas_parameters, filename_input, flow)
    elseif T == EnergyDensityParameters
        filename_input = ifelse(filename == "", "energy_density.txt", filename)
        measurement = EnergyDensityMeasurement(U, meas_parameters, filename_input, flow)
    elseif T == MetaChargeParameters
        filename_input = ifelse(filename == "", "meta_charge.txt", filename)
        measurement = MetaChargeMeasurement(U, meas_parameters, filename_input, false)
    else
        error(T, " is not supported in measurements")
    end

    return measurement
end
