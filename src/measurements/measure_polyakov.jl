struct PolyakovMeasurement{T} <: AbstractMeasurement
    filename::T
    function PolyakovMeasurement(U::Gaugefield; filename="", flow=NoSmearing())
        if is_distributed(U)
            @assert U.topology.numprocs_cart[4] == 1 "Field cannot be decomposed in time direction for polykov loop calculation"
        end

        if !isnothing(filename) && filename != ""
            rpath = StaticString(filename)
            header = ""

            if flow == true || flow != NoSmearing()
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

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                open(filename, "w") do fp
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
    m::PolyakovMeasurement{T},
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    fstr="",
) where {T}
    poly = polyakov_traced(U)
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
        if !isnothing(flow)
            @level1("$itrj\t$(real(poly)) + $(imag(poly))im # poly$(fstr)_$(τ)")
        else
            @level1("$itrj\t$(real(poly)) + $(imag(poly))im # poly")
        end

        if T !== Nothing
            filename = if mpi_multi_sim
                set_ext!(m.filename)
            else
                m.filename
            end

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

function polyakov_traced(U::Gaugefield{B}) where {B}
    @assert U.topology.numprocs_cart[4] == 1 """
    for polyakov loop, the field cannot be partitioned in the t-dimension
    """
    NX, NY, NZ, _ = size(U)
    xrange, yrange, zrange, trange = U.topology.bulk_sites.indices

    P = parallelfor_sum(CartesianIndices((xrange, yrange, zrange)), 0.0+0.0im, B) do p, xyz
        ix, iy, iz = xyz.I
        polymat = U[4, ix, iy, iz, 1]

        for it in trange[2:end]
            polymat = cmatmul_oo(polymat, U[4, ix, iy, iz, it])
        end

        p += tr(polymat)
    end

    return distributed_reduce(P / (NX * NY * NZ), +, U)
end

# TODO:
# function polyakov_traced(U::Gaugefield{CPU,T,true}) where {T}
# end
