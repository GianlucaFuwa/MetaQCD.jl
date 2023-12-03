import ..Gaugefields: plaquette_trace_sum

struct PlaquetteMeasurement{T} <: AbstractMeasurement
    factor::Float64
    fp::T

    function PlaquetteMeasurement(U::Gaugefield; filename="", printvalues=false, flow=false)
        if printvalues
            fp = open(filename, "w")
            header = ""

            if flow
                header *= @sprintf("%-9s\t%-7s\t%-9s\t%-22s",
                                   "itrj", "iflow", "tflow", "Re(plaq)")
            else
                header *= @sprintf("%-9s\t%-22s", "itrj", "Re(plaq)")
            end

            println(fp, header)
        else
            fp = nothing
        end

        factor = 1 / (6*U.NV*U.NC)

        return new{typeof(fp)}(factor, fp)
    end
end

function PlaquetteMeasurement(U, ::PlaquetteParameters, filename, flow=false)
    return PlaquetteMeasurement(U, filename=filename, printvalues=true, flow=flow)
end

function measure(m::PlaquetteMeasurement{T}, U; additional_string="") where {T}
    plaq = plaquette_trace_sum(U) * m.factor
    measurestring = ""

    if T ≡ IOStream
        measurestring *= @sprintf("%-9s\t%-22.15E", additional_string, plaq)
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # plaq"
    end

    output = MeasurementOutput(plaq, measurestring)
    return output
end
