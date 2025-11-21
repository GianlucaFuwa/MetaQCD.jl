struct PlaquetteMeasurement{T} <: AbstractMeasurement
    factor::Float64 # 1 / (6*length(U)*NC)
    filename::T
    function PlaquetteMeasurement(U::Gaugefield; filename="", flow=NoSmearing())
        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            rpath = StaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, ITRJ_STR_FMT, "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, IFLOW_STR_FMT, "iflow")
                    printf(fp, TFLOW_STR_FMT, "tflow")
                end

                printf(fp, METHOD_STR_FMT, "Re(plaq)")
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
        printf(fp, ITRJ_FMT, itrj)

        if !isnothing(flow)
            printf(fp, IFLOW_FMT, iflow)
            printf(fp, TFLOW_FMT, τ)
        end

        printf(fp, METHOD_FMT, plaq)
        newline(fp)
        fclose(fp)
    end

    return plaq
end
