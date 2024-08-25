struct PolyakovMeasurement{T} <: AbstractMeasurement
    filename::T
    function PolyakovMeasurement(::Gaugefield; filename="", flow=false)
        if filename !== nothing && filename != ""
            path = filename * MYEXT
            rpath = StaticString(path)
            header = ""

            if flow
                header *= @sprintf(
                    "%-11s%-7s%-9s%-25s%-25s",
                    "itrj",
                    "iflow",
                    "tflow",
                    "Re(plaq)",
                    "Im(poly)"
                )
            else
                header *= @sprintf("%-11s%-25s%-25s", "itrj", "Re(poly)", "Im(poly)")
            end

            open(path, "w") do fp
                println(fp, header)
            end
        else
            rpath = nothing
        end

        T = typeof(rpath)
        return new{T}(rpath)
    end
end

function PolyakovMeasurement(U, ::PolyakovParameters, filename, flow=false)
    return PolyakovMeasurement(U; filename=filename, flow=flow)
end

function measure(
    ::PolyakovMeasurement{Nothing}, U, ::Integer, itrj, flow=nothing
)
    poly = polyakov_traced(U)

    if mpi_amroot()
        iflow, _ = isnothing(flow) ? (0, 0.0) : flow

        if !isnothing(flow)
            @level1("$itrj\t$(real(poly)) + $(imag(poly))im # poly_flow_$(iflow)")
        else
            @level1("$itrj\t$(real(poly)) + $(imag(poly))im # poly")
        end
    end

    return poly
end

function measure(
    m::PolyakovMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T<:AbstractString}
    poly = polyakov_traced(U)
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    if mpi_amroot()
        filename = set_ext!(m.filename, myinstance)
        fp = fopen(filename, "a")
        printf(fp, "%-11i", itrj)

        if !isnothing(flow)
            printf(fp, "%-7i", iflow)
            printf(fp, "%-9.5f", τ)
        end

        printf(fp, "%-25.15E", real(poly))
        printf(fp, "%-25.15E", imag(poly))
        printf(fp, "\n")
        fclose(fp)
    end

    return poly
end

function polyakov_traced(U::Gaugefield{CPU,T,false}) where {T}
    NX, NY, NZ, NT = global_dims(U)
    P = 0.0 + 0.0im

    @batch reduction = (+, P) for iz in 1:NZ
        for iy in 1:NY
            for ix in 1:NX
                polymat = U[4, ix, iy, iz, 1]

                for it in 1:(NT-1)
                    polymat = cmatmul_oo(polymat, U[4, ix, iy, iz, 1+it])
                end

                P += tr(polymat)
            end
        end
    end

    return P / (NX * NY * NZ)
end

function polyakov_traced(U::Gaugefield{CPU,T,true}) where {T}
    # TODO:
end
