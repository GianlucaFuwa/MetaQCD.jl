mutable struct Polyakov_measurement <: AbstractMeasurement
    filename::Union{Nothing,String}
    verbose_print::Union{Nothing,Verbose_level}
    printvalues::Bool

    function Polyakov_measurement(
        U;
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

        return new(
            filename,
            verbose_print,
            printvalues,
        )
    end
end

function Polyakov_measurement(
    U::Gaugefield,
    params::Poly_parameters,
    filename = "Polyakov.txt")
    return Polyakov_measurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
    )
end

function measure(m::M, U; additional_string = "") where {M <: Polyakov_measurement}
    NX,NY,NZ,_ = size(U)
    poly = polyakov_traced(U) / (NX * NY * NZ)
    measurestring = ""
    if m.printvalues
        measurestring = "$additional_string $(real(poly)) $(imag(poly)) # poly"
        println_verbose2(m.verbose_print, measurestring)
    end
    
    output = Measurement_output(poly, measurestring)
    return output
end

function polyakov_traced(U::Gaugefield)
    space = 8
    poly = zeros(ComplexF64, nthreads()*space)
    NX, NY, NZ, NT = size(U)
    @batch for iz = 1:NZ
        for iy = 1:NY
            for ix = 1:NX
                polymat = U[4][ix,iy,iz,1]
                for t = 1:NT-1
                    polymat *= U[4][ix,iy,iz,1+t]
                end
                poly[threadid()*space] += tr(polymat)
            end
        end
    end
    return sum(poly)
end 