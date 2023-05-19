import ..Gaugefields: gauge_action_wilson, gauge_action_symanzik
import ..Gaugefields: gauge_action_iwasaki, gauge_action_dbw2

mutable struct GaugeActionMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    factor::Float64
    verbose_print::Union{Nothing, VerboseLevel}
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
            if verbose_level == 1
                verbose_print = Verbose1(filename)
            elseif verbose_level == 2
                verbose_print = Verbose2(filename)
            elseif verbose_level == 3
                verbose_print = Verbose3(filename)    
            end
        else
            verbose_print = nothing
        end

        factor = 1 / (U.NV * 6 * U.β)

        return new(
            filename,
            factor,
            verbose_print,
            printvalues,
            GA_methods
        )
    end
end

function GaugeActionMeasurement(
    U::Gaugefield,
    params::ActionParameters,
    filename = "gauge_action.txt",
)
    return GaugeActionMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        GA_methods = params.kinds_of_gauge_action
    )
end

function measure(m::GaugeActionMeasurement, U; additional_string = "")
    measurestring = ""
    values = zeros(Float64, length(m.GA_methods))
    valuedic = Dict{String, AbstractFloat}()
    printstring = additional_string

    for (i, methodname) in enumerate(m.GA_methods)
        if methodname == "wilson"
            Sg_wils = gauge_action_wilson(U) * m.factor
            values[i] = Sg_wils
            valuedic["wilson"] = Sg_wils
        elseif methodname == "symanzik"
            Sg_symanzik = gauge_action_symanzik(U) * m.factor
            values[i] = Sg_symanzik
            valuedic["symanzik"] = Sg_symanzik
        elseif methodname == "iwasaki"
            Sg_iwasaki = gauge_action_symanzik(U) * m.factor
            values[i] = Sg_iwasaki
            valuedic["iwasaki"] = Sg_iwasaki
        elseif methodname == "dbw2"
            Sg_dbw2 = gauge_action_dbw2(U) * m.factor
            values[i] = Sg_dbw2
            valuedic["dbw2"] = Sg_dbw2
        else 
            error("method $methodname is not supported in gauge action measurement")
        end
    end

    if m.printvalues
        for value in values
            printstring *= "$value "
        end

        for methodname in m.TC_methods
            if methodname == "wilson"
                printstring *= "Sg_wilson"
            elseif methodname == "symanzik"
                printstring *= "Sg_symanzik"
            elseif methodname == "iwasaki"
                printstring *= "Sg_iwasaki"
            elseif methodname == "dbw2"
                printstring *= "Sg_dbw2"
            end
        end

        measurestring = printstring
        println_verbose2(m.verbose_print, measurestring)
    end

    output = MeasurementOutput(valuedic, measurestring)
    return output
end