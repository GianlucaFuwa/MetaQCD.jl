struct PolyakovMeasurement{T} <: AbstractMeasurement
    filename::T
    function PolyakovMeasurement(U::Gaugefield; filename="", flow=false)
        if is_distributed(U)
            @assert U.topology.numprocs_cart[4] == 1 "Field cannot be decomposed in time direction for polykov loop calculation"
        end

        if !isnothing(filename) && filename != ""
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

            if mpi_amroot()
                open(path, "w") do fp
                    println(fp, header)
                end
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
    m::PolyakovMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T}
    poly = polyakov_traced(U)
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    if mpi_amroot()
        if !isnothing(flow)
            @level1("$itrj\t$(real(poly)) + $(imag(poly))im # poly_flow_$(τ)")
        else
            @level1("$itrj\t$(real(poly)) + $(imag(poly))im # poly")
        end

        if T !== Nothing
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
    end

    return poly
end

function polyakov_traced(U::Gaugefield{CPU})
    # TODO:
    @assert U.topology.numprocs_cart[4] == 1 "Field cannot be decomposed in time direction for polykov loop calculation"
    NX, NY, NZ, NT = global_dims(U)
    xrange, yrange, zrange, _ = U.topology.bulk_sites.indices
    halo_width = U.topology.halo_width
    P = 0.0 + 0.0im

    @batch reduction = (+, P) for iz in zrange
        for iy in yrange
            for ix in xrange
                polymat = U[4, ix, iy, iz, 1+halo_width]

                for it in 1+halo_width:(NT+halo_width-1)
                    polymat = cmatmul_oo(polymat, U[4, ix, iy, iz, 1+it])
                end

                P += tr(polymat)
            end
        end
    end

    return distributed_reduce(P / (NX * NY * NZ), +, U)
end

# TODO:
# function polyakov_traced(U::Gaugefield{CPU,T,true}) where {T}
# end
