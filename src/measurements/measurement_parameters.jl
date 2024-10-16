function struct2dict(x::T) where {T}
    return Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))
end

abstract type MeasurementParameters end

Base.@kwdef mutable struct EnergyDensityParameters <: MeasurementParameters
    measure_every::Int64 = 10
    kinds_of_energy_density::Vector{String} = ["clover"]
    methodname::String = "energy_density"
end

Base.@kwdef mutable struct GaugeActionParameters <: MeasurementParameters
    measure_every::Int64 = 1
    kinds_of_gauge_action::Vector{String} = ["wilson"]
    methodname::String = "gauge_action"
end

Base.@kwdef mutable struct PlaquetteParameters <: MeasurementParameters
    measure_every::Int64 = 1
    methodname::String = "plaquette"
end

Base.@kwdef mutable struct PolyakovParameters <: MeasurementParameters
    measure_every::Int64 = 1
    methodname::String = "polyakov_loop"
end

Base.@kwdef mutable struct TopologicalChargeParameters <: MeasurementParameters
    measure_every::Int64 = 10
    kinds_of_topological_charge::Vector{String} = ["clover"]
    methodname::String = "topological_charge"
end

Base.@kwdef mutable struct WilsonLoopParameters <: MeasurementParameters
    Tmax::Int64 = 4
    Rmax::Int64 = 4
    measure_every::Int64 = 10
    methodname::String = "wilson_loop"
end

Base.@kwdef mutable struct PionCorrelatorParameters <: MeasurementParameters
    dirac_type::String = "staggered"
    mass::Float64 = 0.1
    r::Float64 = 1.0
    csw::Float64 = 0.0
    eo_precon::Bool = false
    cg_tol::Float64 = 1e-16
    cg_maxiters::Int64 = 1000
    anti_periodic::Bool = true
    measure_every::Int64 = 10
    methodname::String = "pion_correlator"
end

Base.@kwdef mutable struct EigenvaluesParameters <: MeasurementParameters
    dirac_type::String = "staggered"
    mass::Float64 = 0.1
    r::Float64 = 1.0
    csw::Float64 = 0.0
    eo_precon::Bool = false
    anti_periodic::Bool = true
    nev::Int64 = 10
    which::String = "LM"
    mindim::Int64 = max(10, nev)
    maxdim::Int64 = max(20, 2nev)
    tol::Float64 = sqrt(eps(Float64))
    restarts::Int64 = 200
    ddaggerd::Bool = false
    measure_every::Int64 = 10
    methodname::String = "eigenvalues"
end

function initialize_measurement_parameters(methodname)
    if Unicode.normalize(methodname; casefold=true) == "gauge_action"
        method = GaugeActionParameters()
    elseif Unicode.normalize(methodname; casefold=true) == "plaquette"
        method = PlaquetteParameters()
    elseif Unicode.normalize(methodname; casefold=true) == "polyakov_loop"
        method = PolyakovParameters()
    elseif Unicode.normalize(methodname; casefold=true) == "wilson_loop"
        method = WilsonLoopParameters()
    elseif Unicode.normalize(methodname; casefold=true) == "topological_charge"
        method = TopologicalChargeParameters()
    elseif Unicode.normalize(methodname; casefold=true) == "energy_density"
        method = EnergyDensityParameters()
    elseif Unicode.normalize(methodname; casefold=true) == "pion_correlator"
        method = PionCorrelatorParameters()
    elseif Unicode.normalize(methodname; casefold=true) == "eigenvalues"
        method = EigenvaluesParameters()
    else
        error("$methodname is not implemented")
    end
    return method
end

function meas_parameters_from_dict(value_i::Dict)
    methodname = value_i["observable"]
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
    U, meas_parameters::T, filename="", flow=false
) where {T<:MeasurementParameters}
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
    elseif T == PionCorrelatorParameters
        filename_input = ifelse(filename == "", "pion_correlator.txt", filename)
        measurement = PionCorrelatorMeasurement(U, meas_parameters, filename_input, flow)
    elseif T == EigenvaluesParameters
        filename_input = ifelse(filename == "", "eigenvalues.txt", filename)
        measurement = EigenvaluesMeasurement(U, meas_parameters, filename_input, flow)
    else
        error("$T is not supported in measurements")
    end

    return measurement
end
