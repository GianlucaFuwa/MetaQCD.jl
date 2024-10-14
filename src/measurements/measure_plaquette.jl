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

            if mpi_amroot()
                open(path, "w") do fp
                    println(fp, header)
                end
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

function measure(
    m::PlaquetteMeasurement{T}, U, myinstance=mpi_myrank(), itrj=0, flow=nothing
) where {T}
    plaq = plaquette_trace_sum(U) * m.factor
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    if (is_distributed(U) && mpi_amroot()) || !is_distributed(U)
        if !isnothing(flow)
            @level1("$itrj\t$plaq # plaq_flow_$(τ)")
        else
            @level1("$itrj\t$plaq # plaq")
        end

        if T !== Nothing
            filename = set_ext!(m.filename, myinstance)
            fp = fopen(filename, "a")
            printf(fp, "%-11i", itrj)

            if !isnothing(flow)
                printf(fp, "%-7i", iflow)
                printf(fp, "%-9.5f", τ)
            end

            printf(fp, "%+-25.15E", plaq)
            newline(fp)
            fclose(fp)
        end
    end

    return plaq
end
