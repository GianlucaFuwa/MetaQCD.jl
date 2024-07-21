struct GaugeActionMeasurement{T} <: AbstractMeasurement
    GA_dict::Dict{String,Float64} # gauge action definition => value
    factor::Float64 # 1 / (6*U.NV*U.β)
    filename::T
    function GaugeActionMeasurement(U; filename="", GA_methods=["wilson"], flow=false)
        GA_dict = Dict{String,Float64}()
        for method in GA_methods
            @level1("|    Method: $(method)")
            GA_dict[method] = 0.0
        end

        if filename !== nothing && filename != ""
            path = filename * MYEXT
            rpath = StaticString(path)
            header = ""

            if flow
                header *= @sprintf("%-11s%-7s%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-11s", "itrj")
            end

            for methodname in GA_methods
                header *= @sprintf("%-25s", "S_$(methodname)")
            end

            open(path, "w") do fp
                println(fp, header)
            end
        else
            rpath = nothing
        end

        factor = 1 / (6 * U.NV * U.β)
        T = typeof(rpath)
        return new{T}(GA_dict, factor, rpath)
    end
end

function GaugeActionMeasurement(U, params::GaugeActionParameters, filename, flow=false)
    return GaugeActionMeasurement(
        U;
        filename=filename,
        GA_methods=params.kinds_of_gauge_action,
        flow=flow,
    )
end

function measure(
    m::GaugeActionMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T}
    GA_dict = m.GA_dict
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for methodname in keys(GA_dict)
        Sg = calc_gauge_action(U, methodname) * m.factor
        GA_dict[methodname] = Sg
    end

    if T !== Nothing
        filename = set_ext!(m.filename, myinstance)
        fp = fopen(filename, "a")
        printf(fp, "%-11i", itrj::Int64)

        if !isnothing(flow)
            printf(fp, "%-7i", iflow::Int64)
            printf(fp, "%-9.5f", τ::Float64)
        end

        for value in values(GA_dict)
            printf(fp, "%+-25.15E", value::Float64)
        end

        printf(fp, "\n")
        fclose(fp)
    else
        for method in keys(GA_dict)
            S = GA_dict[method]

            if !isnothing(flow)
                @level1("$itrj\t$S # gaction_$(method)_flow_$(iflow)")
            else
                @level1("$itrj\t$S # gaction_$(method)")
            end
        end
    end

    return GA_dict
end
