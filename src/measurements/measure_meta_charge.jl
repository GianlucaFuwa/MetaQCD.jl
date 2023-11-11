struct MetaChargeMeasurement{T} <: AbstractMeasurement
    fp::T

    function MetaChargeMeasurement(::Gaugefield; filename="", printvalues=false)
        if printvalues
            fp = open(filename, "w")
            @printf(fp, "%-9s\t%-22s\n", "itrj", "meta_charge")
        else
            fp = nothing
        end

        return new{typeof(fp)}(fp)
    end
end

function MetaChargeMeasurement(U, ::MetaChargeParameters, filename, ::Bool)
    return MetaChargeMeasurement(U, filename=filename, printvalues=true)
end

function measure(m::MetaChargeMeasurement{T}, U; additional_string="") where {T}
    cv = U.CV
    measurestring = ""

    if T == IOStream
        measurestring *= @sprintf("%-9s\t%+22.15E", additional_string, cv)
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # meta_charge"
    end

    output = MeasurementOutput(cv, measurestring)
    return output
end
