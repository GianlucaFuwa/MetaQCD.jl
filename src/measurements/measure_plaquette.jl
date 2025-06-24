struct PlaquetteMeasurement{T} <: AbstractMeasurement
    factor::Float64 # 1 / (6*length(U)*NC)
    filename::T
    function PlaquetteMeasurement(U::Gaugefield; filename="", flow=NoSmearing())
        if !isnothing(filename) && filename != ""
            rpath = StaticString(filename)
            header = ""

            if flow == true || flow != NoSmearing()
                header *= @sprintf(
                    "%-11s%-7s%-9s%-25s", "itrj", "iflow", "tflow", "Re(plaq)"
                )
            else
                header *= @sprintf("%-11s%-25s", "itrj", "Re(plaq)")
            end

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                open(filename, "w") do fp
                    println(fp, header)
                end
            end
        else
            rpath = nothing
        end

        factor = 1 / 18length(U)
        T = typeof(rpath)
        return new{T}(factor, rpath)
    end
end

function PlaquetteMeasurement(U, ::PlaquetteParameters, filename, flow=false)
    return PlaquetteMeasurement(U; filename=filename, flow=flow)
end

function measure(
    m::PlaquetteMeasurement{T},
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    fstr="",
) where {T}
    plaq = plaquette_trace_sum(U) * m.factor
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
        if !isnothing(flow)
            @level1("$itrj\t$plaq # plaq$(fstr)_$(τ)")
        else
            @level1("$itrj\t$plaq # plaq")
        end

        if T !== Nothing
            filename = if mpi_multi_sim
                set_ext!(m.filename)
            else
                m.filename
            end

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
