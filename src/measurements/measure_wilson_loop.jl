mutable struct Wilson_loop_measurement <: AbstractMeasurement
    filename::Union{Nothing,String}
    verbose_print::Union{Nothing,Verbose_level}
    printvalues::Bool
    Tmax::Int64
    Rmax::Int64
    outputvalues::Matrix{Float64}

    function Wilson_loop_measurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
        Rmax = 4,
        Tmax = 4,
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

        outputvalues = zeros(Float64,Rmax,Tmax)

        return new(
            filename,
            verbose_print,
            printvalues,
            Tmax,
            Rmax,
            outputvalues,
        )
    end
end

function Wilson_loop_measurement(
    params::WilsonLoop_parameters,
    filename = "Wilson_loop.txt")
    return Wilson_loop_measurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        Rmax = params.Rmax,
        Tmax = params.Tmax,
    )
end

function measure(m::M, U; additional_string = "") where {M<:Wilson_loop_measurement}
    measurestring = ""
    for R = 1:m.Rmax
        for T = 1:m.Tmax
            WL = wilsonloop(U, Lμ=R, Lν=T) # NC=3 and 6 associated loop
            m.outputvalues[R,T] = tr(WL) / U.NV / 18.0

            if m.printvalues
                measurestring = " $additional_string $R $T $WL # Wilson_loops # R T W(R,T)"
                println_verbose2(m.verbose_print, measurestring)
            end
        end
    end

    output = Measurement_output(m.outputvalues,measurestring)
    return output
end