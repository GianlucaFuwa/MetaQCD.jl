struct MetaChargeMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    verbose_print::Union{Nothing, VerboseLevel}
    fp::Union{Nothing, IOStream}
    printvalues::Bool

    function MetaChargeMeasurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
    )
        if printvalues
            fp = open(filename, "w")
            header = "itrj\tmeta_charge"

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

        return new(
            filename,
            verbose_print,
            fp,
            printvalues,
        )
    end
end

function MetaChargeMeasurement(
    U::T,
    params::MetaChargeParameters,
    filename,
    ::Bool,
) where {T <: Gaugefield}
    return MetaChargeMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
    )
end

function measure(m::MetaChargeMeasurement, U; additional_string = "")
    cv = U.CV
    measurestring = ""

    if m.printvalues
        cv_str = @sprintf("%.15E", cv)
        measurestring = "$additional_string\t$cv_str\t"
        # println_verbose2(m.verbose_print, "$measurestring# meta_charge")
        println(m.fp, measurestring)
        flush(m.fp)
    end

    output = MeasurementOutput(cv, measurestring * "# meta_charge")
    return output
end
