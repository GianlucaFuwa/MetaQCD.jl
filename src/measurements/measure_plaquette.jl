struct PlaquetteMeasurement{T} <: AbstractMeasurement
    factor::Float64 # 1 / (6*length(U)*NC)
    filename::T
    function PlaquetteMeasurement(U::Gaugefield; filename="", flow=NoSmearing())
        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            rpath = StaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, "%-11s", "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, "%-7s", "iflow")
                    printf(fp, "%-9s", "tflow")
                end

                printf(fp, "%-25s", "Re(plaq)")
                newline(fp)
                fclose(fp)
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

    return plaq
end
