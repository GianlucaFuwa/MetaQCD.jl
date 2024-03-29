struct PolyakovMeasurement{T} <: AbstractMeasurement
    fp::T

    function PolyakovMeasurement(::Gaugefield; filename="", printvalues=false, flow=false)
        if printvalues
            fp = open(filename, "w")
            header = ""

            if flow
                header *= @sprintf("%-9s\t%-7s\t%-9s\t%-22s\t%-22s",
                                   "itrj", "iflow", "tflow", "Re(plaq)", "Im(poly)")
            else
                header *= @sprintf("%-9s\t%-22s\t%-22s", "itrj", "Re(poly)", "Im(poly)")
            end

            println(fp, header)
        else
            fp = nothing
        end

        return new{typeof(fp)}(fp)
    end
end

function PolyakovMeasurement(U, ::PolyakovParameters, filename, flow=false)
    return PolyakovMeasurement(U, filename=filename, printvalues=true, flow=flow)
end

function measure(m::PolyakovMeasurement{T}, U; additional_string="") where {T}
    NX, NY, NZ, _ = size(U)
    poly = polyakov_traced(U) / (NX * NY * NZ)
    measurestring = ""

    if T ≡ IOStream
        measurestring *= @sprintf("%-9s\t%+22.15E\t%+22.15E",
                                  additional_string, real(poly), imag(poly))
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # poly"
    end

    output = MeasurementOutput(poly, measurestring)
    return output
end

function polyakov_traced(U)
    out = zeros(ComplexF64, 8nthreads())
    NX, NY, NZ, NT = size(U)

    @batch per=thread  for iz in 1:NZ
        for iy in 1:NY
            for ix in 1:NX
                polymat = U[4][ix,iy,iz,1]

                for t in 1:NT-1
                    polymat = cmatmul_oo(polymat, U[4][ix,iy,iz,1+t])
                end

                out[8threadid()] += tr(polymat)
            end
        end
    end

    return sum(out)
end
