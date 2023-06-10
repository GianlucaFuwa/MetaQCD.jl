mutable struct PolyakovMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    verbose_print::Union{Nothing, VerboseLevel}
    fp::Union{Nothing, IOStream}
    printvalues::Bool

    function PolyakovMeasurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
    )
        if printvalues
            fp = open(filename, "w")

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

function PolyakovMeasurement(
    U::T,
    params::PolyakovParameters,
    filename = "polyakov.txt",
) where {T<:Gaugefield}
    return PolyakovMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
    )
end

function measure(m::PolyakovMeasurement, U; additional_string = "")
    NX, NY, NZ, _ = size(U)
    poly = polyakov_traced(U) / (NX * NY * NZ)
    measurestring = ""

    if m.printvalues
        measurestring = "$additional_string $(real(poly)) $(imag(poly)) # poly"
        println_verbose2(m.verbose_print, measurestring)
        println(m.fp, measurestring)
        flush(m.fp)
    end

    output = MeasurementOutput(poly, measurestring)
    return output
end

function polyakov_traced(U::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(U)
    spacing = 8
    poly = zeros(ComplexF64, nthreads() * spacing)

    @batch for iz in 1:NZ
        for iy in 1:NY
            for ix in 1:NX
                polymat = U[4][ix,iy,iz,1]

                for t in 1:NT-1
                    polymat = cmatmul_oo(polymat, U[4][ix,iy,iz,1+t])
                end

                poly[threadid() * spacing] += tr(polymat)
            end
        end
    end

    return sum(poly)
end
