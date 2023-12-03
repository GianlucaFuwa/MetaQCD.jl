import ..Gaugefields: WilsonGaugeAction, SymanzikTreeGaugeAction, SymanzikTadGaugeAction,
    IwasakiGaugeAction, DBW2GaugeAction

struct GaugeActionMeasurement{T} <: AbstractMeasurement
    GA_dict::Dict{String, Float64}
    factor::Float64
    fp::T

    function GaugeActionMeasurement(
        U;
        filename = "",
        printvalues = false,
        GA_methods = ["wilson"],
        flow = false,
    )
        GA_dict = Dict{String, Float64}()
        for method in GA_methods
            GA_dict[method] = 0.0
        end

        if printvalues
            fp = open(filename, "w")
            header = ""
            if flow
                header *= @sprintf("%-9s\t%-7s\t%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-9s", "itrj")
            end

            for methodname in GA_methods
                header *= @sprintf("\t%-22s", "S_$(methodname)")
            end

            println(fp, header)
        else
            fp = nothing
        end

        factor = 1 / (6*U.NV*U.β)

        return new{typeof(fp)}(GA_dict, factor, fp)
    end
end

function GaugeActionMeasurement(U, params::GaugeActionParameters, filename, flow=false)
    return GaugeActionMeasurement(U, filename = filename, printvalues = true,
                                  GA_methods = params.kinds_of_gauge_action, flow = flow)
end

function measure(m::GaugeActionMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)

    for methodname in keys(m.GA_dict)
        Sg = calc_gauge_action(U, methodname) * m.factor
        m.GA_dict[methodname] = Sg
    end

    if T == IOStream
        for value in values(m.GA_dict)
            svalue = @sprintf("%-22.15E", value)
            printstring *= "\t$svalue"
        end

        measurestring = printstring
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # gaction"
    end

    output = MeasurementOutput(m.GA_dict, measurestring)
    return output
end
