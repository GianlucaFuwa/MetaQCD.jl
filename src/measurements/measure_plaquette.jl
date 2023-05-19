import ..Gaugefields: plaquette_trace_sum

mutable struct PlaquetteMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    factor::Float64
    verbose_print::Union{Nothing, VerboseLevel}
    printvalues::Bool

    function PlaquetteMeasurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
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
        factor = 1 / (6 * U.NV * U.NC)

        return new(
            filename,
            factor,
            verbose_print,
            printvalues,
        )
    end
end

function PlaquetteMeasurement(U::Gaugefield, params::PlaquetteParameters, filename)
    return PlaquetteMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
    )
end

function measure(m::PlaquetteMeasurement, U; additional_string = "")
    plaq = plaquette_trace_sum(U) * m.factor
    measurestring = ""
    
    if m.printvalues
        measurestring = "$additional_string $plaq # plaq"
        println_verbose2(m.verbose_print, measurestring)
    end

    output = MeasurementOutput(plaq, measurestring)
    return output
end