struct PolyakovMeasurement{T} <: AbstractMeasurement
    filename::T
    function PolyakovMeasurement(U::Gaugefield; filename="", flow=NoSmearing())
        if is_distributed(U)
            @assert U.topology.numprocs_cart[4] == 1 "Field cannot be decomposed in time direction for polykov loop calculation"
        end

        if !isnothing(filename) && filename != ""
            rpath = StaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, "%-11s", "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, "%-7s", "iflow")
                    printf(fp, "%-9s", "tflow")
                end

                printf(fp, "%-25s", "Re(poly)")
                printf(fp, "%-25s", "Im(poly)")
                newline(fp)
                fclose(fp)
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

function polyakov_traced(U::Gaugefield{B,T,M}) where {B,T,M}
    @assert U.topology.numprocs_cart[4] == 1 """
    for polyakov loop, the field cannot be partitioned in the t-dimension
    """
    NX, NY, NZ, _ = size(U)
    xrange, yrange, zrange, trange = U.topology.bulk_sites.indices
    itr = CartesianIndices((xrange, yrange, zrange))
    P = parallelfor_sum(itr, 0.0+0.0im, B, Val(M), (), (), (U,)) do p, xyz, U
        ix, iy, iz = xyz.I
        polymat = U[4, ix, iy, iz, 1]

        for it in trange[2:end]
            polymat = cmatmul_oo(polymat, U[4, ix, iy, iz, it])
        end

        p += tr(polymat)
    end

    return distributed_reduce(P / (NX * NY * NZ), +, U)
end
