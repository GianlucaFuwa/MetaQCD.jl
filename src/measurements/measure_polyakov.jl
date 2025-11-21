struct PolyakovMeasurement{T} <: AbstractMeasurement
    filename::T
    function PolyakovMeasurement(U::Gaugefield; filename="", flow=NoSmearing())
        if is_distributed(U)
            @assert U.topology.numprocs_cart[4] == 1 "Field cannot be decomposed in time direction for polykov loop calculation"
        end

        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            rpath = StaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, ITRJ_STR_FMT, "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, IFLOW_STR_FMT, "iflow")
                    printf(fp, TFLOW_STR_FMT, "tflow")
                end

                printf(fp, METHOD_STR_FMT, "Re(poly)")
                printf(fp, METHOD_STR_FMT, "Im(poly)")
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
        printf(fp, ITRJ_FMT, itrj)

        if !isnothing(flow)
            printf(fp, IFLOW_FMT, iflow)
            printf(fp, TFLOW_FMT, τ)
        end

        printf(fp, METHOD_FMT, real(poly))
        printf(fp, METHOD_FMT, imag(poly))
        printf(fp, "\n")
        fclose(fp)
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
    P = parallelfor_sum(itr, 0.0+0.0im, B, Val(M), (), (), (U,)) do p, xyz, (U,)
        ix, iy, iz = xyz.I
        @inbounds polymat = U[4, CartesianIndex(ix, iy, iz, 1)]

        for it in trange[2:end]
            @inbounds polymat = cmatmul_oo(polymat, U[4, CartesianIndex(ix, iy, iz, it)])
        end

        p += tr(polymat)
    end

    return distributed_reduce(P / (NX * NY * NZ), +, U)
end
