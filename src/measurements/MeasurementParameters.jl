struct2dict(x::T) where {T} = 
    Dict{String,Any}(string(fn) => getfield(x, fn) for fn ∈ fieldnames(T))

abstract type Measurement_parameters end

Base.@kwdef mutable struct Action_parameters <: Measurement_parameters
    methodname::String = "Action"
    kind_of_gaction::String = "Plaquette"
    Verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
end

Base.@kwdef mutable struct Plaq_parameters <: Measurement_parameters
    methodname::String = "Plaquette"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
end

Base.@kwdef mutable struct Poly_parameters <: Measurement_parameters
    methodname::String = "Polyakov_loop"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 1
end

Base.@kwdef mutable struct WilsonLoop_parameters <: Measurement_parameters
    methodname::String = "Wilson_loop"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    Tmax::Int64 = 4
    Rmax::Int64 = 4
    measure_every::Int64 = 10
end

Base.@kwdef mutable struct TopCharge_parameters <: Measurement_parameters
    methodname::String = "Topological_charge"
    verbose_level::Int64 = 2
    printvalues::Bool = true
    measure_every::Int64 = 10
    kinds_of_topological_charge::Vector{String} = ["clover"]
end

function initialize_measurement_parameters(methodname)
    if methodname == "Action"
        method = Action_parameters()
    elseif methodname == "Plaquette"
        method = Plaq_parameters()
    elseif methodname == "Polyakov_loop"
        method = Poly_parameters()
    elseif methodname == "Wilson_loop"
        method = WilsonLoop_parameters()
    elseif methodname == "Topological_charge"
        method = TopCharge_parameters()
    else 
        error("$methodname is not implemented")
    end
    return method
end

function prepare_measurement_from_dict(U, value_i::Dict, filename = "")
    parameters = construct_Measurement_parameters_from_dict(value_i)
    return prepare_measurement(U, parameters, filename)
end

function construct_Measurement_parameters_from_dict(value_i::Dict)
    @assert haskey(value_i, "methodname") "methodname should be in measurement."
    methodname = value_i["methodname"]
    method = initialize_measurement_parameters(methodname)
    method_dict = struct2dict(method)
    for (key_ii, value_ii) in value_i
        if haskey(method_dict,key_ii)
            if typeof(value_ii) !== nothing
                keytype = typeof(getfield(method, Symbol(key_ii)))
                setfield!(method, Symbol(key_ii), keytype(value_ii))
            end
        end
    end
    value_out = deepcopy(method)
    return value_out
end

function prepare_measurement(U, measurement_parameters::T, filename="") where {T}
    if T == Action_parameters
        filename_input = ifelse(filename == "", "Action.txt",filename)
        measurement = Action_measurement(U, measurement_parameters, filename_input)
    elseif T == Plaq_parameters
        filename_input = ifelse(filename == "", "Plaquette.txt", filename)
        measurement = Plaquette_measurement(U, measurement_parameters, filename_input)
    elseif T == Poly_parameters
        filename_input = ifelse(filename == "", "Polyakov_loop.txt", filename)
        measurement = Polyakov_measurement(U, measurement_parameters, filename_input)
    elseif T == TopCharge_parameters
        filename_input = ifelse(filename == "", "Topological_charge.txt", filename)
        measurement = Topological_charge_measurement(U, measurement_parameters, filename_input)
    elseif T == WilsonLoop_parameters
        filename_input = ifelse(filename == "", "Wilson_loop.txt", filename)
        measurement = Wilson_loop_measurement(U, measurement_parameters, filename_input)
    else
        error(T, " is not supported in measurements")
    end
    return measurement
end
