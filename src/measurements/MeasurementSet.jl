function set_parameter_default(method, key, defaultvalue)
    if haskey(method, key)
        return method[key]
    else
        return defaultvalue
    end
end

struct MeasurementSet
    nummeasurements::Int64
    measurements::Vector{AbstractMeasurement}
    measurement_methods::Vector{Dict}
    methodnames::Vector{String}
    baremeasurement_indices::Vector{Int64}
    flowmeasurement_indices::Vector{Int64}
end

function measure(m::MeasurementSet, itrj, U)
    additional_string = "$itrj 0 0.0 "

    for i in m.baremesurement_indices
        measure(m.measurements[i], U, additional_string = additional_string)
    end

end

function measure_withflow(
    m::MeasurementSet,
    itrj,
    Uin,
    smearing::T,
    numstep,
    dτ,
) where {T<:GradientFlow}
    Usmr = smearing.Usmr

    for istep = 1:numstep
        τ = istep * dτ
        flow!(Usmr, smearing)
        additional_string = "$itrj $istep $τ "
        
        for i in m.flowmeasurement_indices
            measure(
                m.measurements[i],
                smearing.Usmeared_multi[i],
                additional_string = additional_string,
            )
        end
    end

    return nothing
end