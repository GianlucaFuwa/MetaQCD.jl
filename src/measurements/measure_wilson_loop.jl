import ..Gaugefields: wilsonloop

struct WilsonLoopMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    verbose_print::Union{Nothing, VerboseLevel}
    fp::Union{Nothing, IOStream}
    printvalues::Bool
    Tmax::Int64
    Rmax::Int64
    outputvalues::Matrix{Float64}

    function WilsonLoopMeasurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
        Rmax = 4,
        Tmax = 4,
        flow = false,
    )
        if printvalues
            fp = open(filename, "w")
            header = ""

            if flow
                header *= "itrj\tiflow\ttflow\tRe(wilson_loop)\t(in column major)"
            else
                "itrj\tRe(wilson_loop)\t(in column major)"
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

        outputvalues = zeros(Float64, Rmax, Tmax)

        return new(
            filename,
            verbose_print,
            fp,
            printvalues,
            Tmax,
            Rmax,
            outputvalues,
        )
    end
end

function WilsonLoopMeasurement(
    U::T,
    params::WilsonLoopParameters,
    filename = "wilson_loop.txt",
    flow = false,
) where {T <: Gaugefield}
    return WilsonLoopMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        Rmax = params.Rmax,
        Tmax = params.Tmax,
        flow = flow,
    )
end

function measure(m::WilsonLoopMeasurement, U; additional_string = "")
    measurestring = ""

    for T in 1:m.Tmax
        for R in 1:m.Rmax
            WL = wilsonloop(U, R, T) # NC=3 and 6 associated loop
            m.outputvalues[R, T] = tr(WL) / (U.NV * 18.0)

            if m.printvalues
                measurestring = "$additional_string\t$R\t$T\t$WL"
                # println_verbose2(m.verbose_print, "$measurestring# wilson_loops")
                println(m.fp, measurestring)
                flush(m.fp)
            end
        end
    end

    output = MeasurementOutput(m.outputvalues, measurestring * " # wilson_loops")
    return output
end
