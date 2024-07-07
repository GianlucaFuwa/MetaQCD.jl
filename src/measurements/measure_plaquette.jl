struct PlaquetteMeasurement{T} <: AbstractMeasurement
    factor::Float64 # 1 / (6*U.NV*U.NC)
    filename::T
    myinstance::Base.RefValue{Int64}
    function PlaquetteMeasurement(
        U::Gaugefield; filename::Union{String,Nothing}=nothing, flow=false, printvalues=false
    )
        if filename !== nothing
            @assert filename != ""
            header = ""

            if flow
                header *= @sprintf(
                    "%-9s\t%-7s\t%-9s\t%-22s", "itrj", "iflow", "tflow", "Re(plaq)"
                )
            else
                header *= @sprintf("%-9s\t%-22s", "itrj", "Re(plaq)")
            end

            open(filename * "_$MYRANK", "w") do io
                println(io, header)
            end
        end

        factor = 1 / (6 * U.NV * U.NC)

        T = typeof(filename)
        return new{T}(factor, filename, Base.RefValue{Int64}(MYRANK))
    end
end

function PlaquetteMeasurement(U, ::PlaquetteParameters, filename, flow=false)
    return PlaquetteMeasurement(U; filename=filename, printvalues=true, flow=flow)
end

function measure(m::PlaquetteMeasurement{T}, U; additional_string="") where {T}
    plaq = plaquette_trace_sum(U) * m.factor
    measurestring = ""

    if T !== Nothing
        measurestring *= @sprintf("%-9s\t%-22.15E", additional_string, plaq)
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # plaq"
    end

    output = MeasurementOutput(plaq, measurestring)
    return output
end
