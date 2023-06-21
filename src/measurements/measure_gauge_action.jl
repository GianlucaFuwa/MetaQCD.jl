import ..Gaugefields: WilsonGaugeAction, SymanzikTreeGaugeAction, SymanzikTadGaugeAction,
    IwasakiGaugeAction, DBW2GaugeAction

mutable struct GaugeActionMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    factor::Float64
    verbose_print::Union{Nothing, VerboseLevel}
    fp::Union{Nothing, IOStream}
    printvalues::Bool
    GA_methods::Vector{String}

    function GaugeActionMeasurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
        GA_methods = ["wilson"]
    )
        if printvalues
            fp = open(filename, "w")
            header = ""
            header *= "itrj"

            for methodname in GA_methods
                header *= "\t$methodname"
            end

            println(fp, header)

            if verbose_level == 1
                verbose_print = Verbose1()
            elseif verbose_level == 2
                verbose_print = Verbose2()
            elseif verbose_level == 3
                verbose_print = Verbose3()
            end
        else
            fp = nothing
            verbose_print = nothing
        end

        factor = 1 / (U.NV * 6 * U.β)

        return new(
            filename,
            factor,
            verbose_print,
            fp,
            printvalues,
            GA_methods,
        )
    end
end

function GaugeActionMeasurement(
    U::T,
    params::GaugeActionParameters,
    filename = "gauge_action.txt",
) where {T <: Gaugefield}
    return GaugeActionMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        GA_methods = params.kinds_of_gaction,
    )
end

function measure(m::GaugeActionMeasurement, U; additional_string = "")
    measurestring = ""
    values = zeros(Float64, length(m.GA_methods))
    valuedic = Dict{String, AbstractFloat}()
    printstring = "$additional_string\t"

    for (i, methodname) in enumerate(m.GA_methods)
        if methodname == "wilson"
            Sg_wils = WilsonGaugeAction()(U) * m.factor
            values[i] = Sg_wils
            valuedic["wilson"] = Sg_wils
        elseif methodname == "symanzik_tree"
            Sg_symanzik_tree = SymanzikTreeGaugeAction()(U) * m.factor
            values[i] = Sg_symanzik_tree
            valuedic["symanzik_tree"] = Sg_symanzik_tree
        elseif methodname == "symanzik_tad"
            Sg_symanzik_tad = SymanzikTadGaugeAction()(U) * m.factor
            values[i] = Sg_symanzik_tad
            valuedic["symanzik_tad"] = Sg_symanzik_tad
        elseif methodname == "iwasaki"
            Sg_iwasaki = IwasakiGaugeAction()(U) * m.factor
            values[i] = Sg_iwasaki
            valuedic["iwasaki"] = Sg_iwasaki
        elseif methodname == "dbw2"
            Sg_dbw2 = DBW2GaugeAction()(U) * m.factor
            values[i] = Sg_dbw2
            valuedic["dbw2"] = Sg_dbw2
        else
            error("method $methodname is not supported in gauge action measurement")
        end
    end

    if m.printvalues
        for value in values
            printstring *= "$value\t"
        end

        measurestring = printstring
        println_verbose2(m.verbose_print, measurestring, "# gaction")
        println(m.fp, measurestring)
        flush(m.fp)
    end

    output = MeasurementOutput(valuedic, measurestring)
    return output
end
