import .AbstractMeasurementModule: AbstractMeasurement, EnergyDensityMeasurement,
    GaugeActionMeasurement, PlaquetteMeasurement, PolyakovMeasurement,
    TopologicalChargeMeasurement, WilsonLoopMeasurement, MetaChargeMeasurement,
    MetaChargeParameters, MeasurementParameters
import .AbstractMeasurementModule: construct_measurement_parameters_from_dict,
    prepare_measurement, get_string, get_value, measure
import .AbstractSmearingModule: flow!
import .Gaugefields: substitute_U!


struct MeasurementMethods
    measurement_parameters_set::Vector{MeasurementParameters}
    measurements::Vector{AbstractMeasurement}
    num_measurements::Int64
    intervals::Vector{Int64}
end

function MeasurementMethods(
    U,
    measurement_dir,
    measurement_methods::Vector{Dict};
    flow = false,
    cv = false,
    additional_string = "",
)
    nummeasurements = length(measurement_methods) + cv
    measurements = Vector{AbstractMeasurement}(undef, nummeasurements + cv)
    measurement_parameters_set = Vector{MeasurementParameters}(undef, nummeasurements + cv)
    intervals = zeros(Int64, nummeasurements + cv)

    str = flow ? "_flowed" : ""
    idx = 1

    for (i, method) in enumerate(measurement_methods)
        measurement_parameters = construct_measurement_parameters_from_dict(method)
        intervals[i] = measurement_parameters.measure_every
        filename = measurement_dir * "/" *
            measurement_parameters.methodname * additional_string * "$str.txt"
        measurements[i] = prepare_measurement(U, measurement_parameters, filename, flow)
        measurement_parameters_set[i] = deepcopy(measurement_parameters)
        idx += 1
    end

    if cv == true && flow == false
        measurement_parameters = MetaChargeParameters()
        intervals[idx] = measurement_parameters.measure_every
        filename = measurement_dir * "/meta_charge" * additional_string * ".txt"
        measurements[idx] = prepare_measurement(U, measurement_parameters, filename, false)
        measurement_parameters_set[idx] = deepcopy(measurement_parameters)
    end

    return MeasurementMethods(
        measurement_parameters_set,
        measurements,
        nummeasurements,
        intervals,
    )
end

function calc_measurements(
    m::MeasurementMethods,
    itrj,
    U;
    additional_string = "",
)
    measurestrings = String[]
    check_for_measurements(itrj, m.intervals) || return measurestrings

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

function calc_measurements_flowed(
    m::MeasurementMethods,
    gradient_flow,
    itrj,
    U;
)
    measurestrings = String[]
    check_for_measurements(itrj, m.intervals) || return measurestrings
    substitute_U!(gradient_flow.Uflow, U)

    for iflow in 1:gradient_flow.numflow
        τ = round(iflow * gradient_flow.ϵ * gradient_flow.steps, sigdigits = 3)
        flow!(gradient_flow)
        additional_string = "$itrj\t$iflow\t$τ"

        if iflow % gradient_flow.measure_every == 0
            for i in 1:m.num_measurements
                interval = m.intervals[i]

                if itrj % interval == 0
                    outputvalue = measure(
                        m.measurements[i],
                        gradient_flow.Uflow,
                        additional_string = additional_string,
                    )
                    push!(measurestrings, get_string(outputvalue))
                end

            end
        end

    end

    return measurestrings
end

function check_for_measurements(itrj, intervals)
    for num in intervals
        (itrj % num == 0) && return true
    end

    return false
end

function Base.close(m::MeasurementMethods)
    for meas in m.measurements
        println(meas)
        if meas.fp !== nothing
            close(meas.fp)
        end
    end

    return nothing
end
