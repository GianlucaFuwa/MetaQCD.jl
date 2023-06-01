import .AbstractMeasurementModule: construct_measurement_parameters_from_dict
import .AbstractMeasurementModule: MeasurementParameters, prepare_measurement
import .AbstractMeasurementModule: get_string, get_value, measure
import .AbstractMeasurementModule: AbstractMeasurement, GaugeActionMeasurement
import .AbstractMeasurementModule: PlaquetteMeasurement, PolyakovMeasurement
import .AbstractMeasurementModule: TopologicalChargeMeasurement, WilsonLoopMeasurement

struct MeasurementMethods
    measurement_parameters_set::Vector{MeasurementParameters}
    measurements::Vector{AbstractMeasurement}
    num_measurements::Int64
    intervals::Vector{Int64}
end

function calc_measurement_values(m::MeasurementMethods, itrj, U; additional_string = "")
    measurestrings = String[]

    for i in 1:m.num_measurements
        interval = m.intervals[i]
        if itrj % interval == 0
            outputvalue = measure(
                m.measurements[i],
                U,
                additional_string = "$itrj " * additional_string,
            )
            push!(measurestrings, get_string(outputvalue))
        end
    end

    return measurestrings
end

function MeasurementMethods(U, measurement_dir, measurement_methods::Vector{Dict})
    nummeasurements = length(measurement_methods)
    measurements = Vector{AbstractMeasurement}(undef, nummeasurements)
    measurement_parameters_set = Vector{MeasurementParameters}(undef, nummeasurements)
    intervals = zeros(Int64, nummeasurements)

    for (i, method) in enumerate(measurement_methods)
        measurement_parameters = construct_measurement_parameters_from_dict(method)
        intervals[i] = measurement_parameters.measure_every
        filename = measurement_dir * "/" * measurement_parameters.methodname * ".txt"
        measurements[i] = prepare_measurement(U, measurement_parameters, filename)
        measurement_parameters_set[i] = deepcopy(measurement_parameters)
    end
    
    return MeasurementMethods(
        measurement_parameters_set,
        measurements,
        nummeasurements,
        intervals,
    )
end