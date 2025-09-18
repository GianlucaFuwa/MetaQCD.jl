struct GaugeActionMeasurement{T} <: AbstractMeasurement
    GA_dict::Dict{String,Float64} # gauge action definition => value
    factor::Float64 # 1 / (6*length(U)*U.β)
    filename::T
    function GaugeActionMeasurement(
        U; filename="", GA_methods=["wilson"], flow=NoSmearing()
    )
        GA_dict = Dict{String,Float64}()

        for method in GA_methods
            @level1("|    type: $(method)")
            GA_dict[method] = 0.0
        end

        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            rpath = StaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, "%-11s", "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, "%-7s", "iflow")
                    printf(fp, "%-9s", "tflow")
                end

                for method in keys(GA_dict)
                    printf(fp, "%-25s", "S_$(method)")
                end

                newline(fp)
                fclose(fp)
            end
        else
            rpath = nothing
        end

        factor = 1 / (6 * length(U) * U.β)
        T = typeof(rpath)
        return new{T}(GA_dict, factor, rpath)
    end
end

function GaugeActionMeasurement(U, params::GaugeActionParameters, filename, flow=false)
    return GaugeActionMeasurement(
        U;
        filename=filename,
        GA_methods=params.type,
        flow=flow,
    )
end

function measure(
    m::GaugeActionMeasurement{T},
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    fstr="",
) where {T}
    GA_dict = m.GA_dict
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(GA_dict)
        GA_dict[method] = calc_gauge_action(U, method) * m.factor
    end

    for method in keys(GA_dict)
        S = GA_dict[method]

        if !isnothing(flow)
            @level1("$itrj\t$S # gaction_$(method)$(fstr)_$(τ)")
        else
            @level1("$itrj\t$S # gaction_$(method)")
        end
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

        for method in keys(GA_dict)
            printf(fp, "%+-25.15E", GA_dict[method])
        end

        printf(fp, "\n")
        fclose(fp)
    end

    return GA_dict
end
