mutable struct Plaquette_measurement <: AbstractMeasurement
    filename::Union{Nothing,String}
    factor::Float64
    verbose_print::Union{Nothing,Verbose_level}
    printvalues::Bool

    function Plaquette_measurement(
        U::Gaugefield;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
    )
        if printvalues
            if verbose_level == 1
                verbose_print = Verbose_1(filename)
            elseif verbose_level == 2
                verbose_print = Verbose_2(filename)
            elseif verbose_level == 3
                verbose_print = Verbose_3(filename)    
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

function Plaquette_measurement(U::Gaugefield, params::Plaq_parameters, filename)
    return Plaquette_measurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
    )
end

function measure(m::M, U; additional_string = "") where {M<:Plaquette_measurement}
    plaq = plaquette_tracedsum(U) * m.factor
    measurestring = ""
    if m.printvalues
        measurestring = "$additional_string $plaq # plaq"
        println_verbose2(m.verbose_print, measurestring)
    end

    output = Measurement_output(plaq, measurestring)
    return output
end