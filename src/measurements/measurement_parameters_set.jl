import .AbstractMeasurement_module:
    construct_Measurement_parameters_from_dict,
    Measurement_parameters,
    AbstractMeasurement,
    prepare_measurement,
    get_string,
    Plaquette_measurement,
    measure,
    get_value,
    Polyakov_measurement,
    Topological_charge_measurement,
    Wilson_loop_measurement,
    Gauge_action_measurement

struct Measurement_methods
    measurement_parameters_set::Vector{Measurement_parameters}
    measurements::Vector{AbstractMeasurement}
    num_measurements::Int64
    intervals::Vector{Int64}
end

function calc_measurement_values(m::Measurement_methods, itrj, U; additional_string = "")
    measurestrings = String[]
    for i = 1:m.num_measurements
        interval = m.intervals[i]
        if itrj % interval == 0
            outputvalue = measure(
                m.measurements[i],
                U,
                additional_string = "itrj " * additional_string,
            )
            push!(measurestrings, get_string(outputvalue))
        end
    end
    return measurestrings
end

function Measurement_methods(
    U,
    measurement_dir,
    measurement_methods::T,
) where {T<:Vector{Dict}}
    nummeasurements = length(measurement_methods)
    measurements = Vector{AbstractMeasurement}(undef, nummeasurements)
    measurement_parameters_set = Vector{Measurement_parameters}(undef, nummeasurements)
    intervals = zeros(Int64, nummeasurements)

    for (i, method) in enumerate(measurement_methods)
        measurement_parameters = construct_Measurement_parameters_from_dict(method)
        intervals[i] = measurement_parameters.measure_every
        filename = measurement_dir * "/" * measurement_parameters.methodname * ".txt"
        meaasurements[i] = prepare_measurement(U, measurement_parameters, filename)
        measurement_parameters_set[i] = deepcopy(measurement_parameters)
    end
    
    return Measurement_methods(
        measurement_parameters_set,
        measurements,
        nummeasurements,
        intervals,
    )
end

