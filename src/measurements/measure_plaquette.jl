struct PlaquetteMeasurement{T} <: AbstractMeasurement
    factor::Float64 # 1 / (6*U.NV*U.NC)
    filename::T
    function PlaquetteMeasurement(U::Gaugefield; filename="", flow=false)
        if !isnothing(filename) && filename != ""
            path = filename * MYEXT
            rpath = StaticString(path)
            header = ""

            if flow
                header *= @sprintf(
                    "%-11s%-7s%-9s%-25s", "itrj", "iflow", "tflow", "Re(plaq)"
                )
            else
                header *= @sprintf("%-11s%-25s", "itrj", "Re(plaq)")
            end

            open(path, "w") do fp
                println(fp, header)
            end
        else
            rpath = nothing
        end

        factor = 1 / (6 * U.NV * U.NC)
        T = typeof(rpath)
        return new{T}(factor, rpath)
    end
end

function PlaquetteMeasurement(U, ::PlaquetteParameters, filename, flow=false)
    return PlaquetteMeasurement(U; filename=filename, flow=flow)
end

@inline function measure(
    m::PlaquetteMeasurement{Nothing}, U, ::Integer, itrj, flow=nothing
)
    plaq = plaquette_trace_sum(U) * m.factor
    iflow, _ = isnothing(flow) ? (0, 0.0) : flow

    if !isnothing(flow)
        @level1("$itrj\t$plaq # plaq_flow_$(iflow)")
    else
        @level1("$itrj\t$plaq # plaq")
    end
    return plaq
end

@inline function measure(
    m::PlaquetteMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T<:AbstractString}
    plaq = plaquette_trace_sum(U) * m.factor
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    filename = set_ext!(m.filename, myinstance)
    fp = fopen(filename, "a")
    printf(fp, "%-11i", itrj)

    if !isnothing(flow)
        printf(fp, "%-7i", iflow)
        printf(fp, "%-9.5f", τ)
    end

    printf(fp, "%+-25.15E\n", plaq)
    fclose(fp)
    return plaq
end
