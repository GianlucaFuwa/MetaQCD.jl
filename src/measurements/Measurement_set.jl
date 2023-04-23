import ..Abstractsmearing

function set_parameter_default(method, key, defaultvalue)
    if haskey(method, key)
        return method[key]
    else
        return defaultvalue
    end
end

struct Measurements_set
    nummeasurements::Int64
    measurements::Vector{AbstractMeasurement}
    measurement_methods::Vector{Dict}
    methodnames::Vector{String}
    baremeasurement_indices::Vector{Int64}
    
end