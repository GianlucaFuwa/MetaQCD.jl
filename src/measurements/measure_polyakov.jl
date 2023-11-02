mutable struct PolyakovMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    verbose_print::Union{Nothing, VerboseLevel}
    fp::Union{Nothing, IOStream}
    printvalues::Bool

    function PolyakovMeasurement(U; filename=nothing, verbose_level=2, printvalues=false, flow=false)
        if printvalues
            fp = open(filename, "w")
            header = ""

            if flow
                header *= "itrj\tiflow\ttflow\t$(rpad("Re(poly)", 22, " "))" *
                    "\t$(rpad("Im(poly)", 22, " "))"
            else
                header *= "$(rpad("itrj", 9, " "))\t$(rpad("Re(poly)", 22, " "))" *
                    "\t$(rpad("Im(poly)", 22, " "))"
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

        return new(filename, verbose_print, fp, printvalues)
    end
end

function PolyakovMeasurement(U, params::PolyakovParameters, filename, flow=false)
    return PolyakovMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        flow = flow,
    )
end

function measure(m::PolyakovMeasurement, U; additional_string="")
    NX, NY, NZ, _ = size(U)
    poly = polyakov_traced(U) / (NX * NY * NZ)
    measurestring = ""

    if m.printvalues
        poly_re = @sprintf("%.15E", real(poly))
        poly_im = @sprintf("%.15E", imag(poly))
        measurestring = "$(rpad(additional_string, 9, " "))\t$poly_re\t$poly_im"
        # println_verbose2(m.verbose_print, "$measurestring# poly")
        println(m.fp, measurestring)
        flush(m.fp)
    end

    output = MeasurementOutput(poly, measurestring * " # poly")
    return output
end

function polyakov_traced(U)
    NX, NY, NZ, NT = size(U)

    @batch threadlocal=zero(ComplexF64)::ComplexF64 for iz in 1:NZ
        for iy in 1:NY
            for ix in 1:NX
                polymat = U[4][ix,iy,iz,1]

                for t in 1:NT-1
                    polymat = cmatmul_oo(polymat, U[4][ix,iy,iz,1+t])
                end

                threadlocal += tr(polymat)
            end
        end
    end

    return sum(threadlocal)
end
