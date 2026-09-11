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
        @level2("$itrj\t$(real(poly)) + $(imag(poly))im # poly$(fstr)_$(τ)")
    else
        @level2("$itrj\t$(real(poly)) + $(imag(poly))im # poly")
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
        L = polyakov_loop_kernel(U, ix, iy, iz, trange)
        p += tr(L)
    end

    return distributed_reduce(P / (NX * NY * NZ), +, U)
    # return distributed_reduce(P, +, U)
end

@inline function polyakov_loop_kernel(U, ix, iy, iz, trange)
    @inbounds L = U[4, CartesianIndex(ix, iy, iz, 1)]

    for it in trange[2:end]
        @inbounds L = cmatmul_oo(L, U[4, CartesianIndex(ix, iy, iz, it)])
    end

    return L
end

function polyakov_deriv!(
    dU::Colorfield{B,T,M}, U::Gaugefield{B,TU}, deriv_fac, fac=1.0
) where {B,T,M,TU}
    @assert U.topology.numprocs_cart[4] == 1 """
    for polyakov loop, the field cannot be partitioned in the t-dimension
    """
    clear!(dU) # set all to 0
    NX, NY, NZ, NT = size(U)
    c = T(fac / 2 / (NX * NY * NZ))

    parallelfor(eachindex(dU, U), B, Val(M), (), (U,), (dU, U)) do site, (dU, U)
        ix, iy, iz, it = site.I
        tmp = U[4, site]

        for j in it+1:NT 
            tmp = cmatmul_oo(tmp, U[4, CartesianIndex(ix, iy, iz, j)])
        end

        for k in 1:it-1
            tmp = cmatmul_oo(tmp, U[4, CartesianIndex(ix, iy, iz, k)])
        end

        dU[4, site] = c * traceless_antihermitian(deriv_fac * tmp)
    end

    return nothing
end

function polyakov_mag_deriv!(
    dU::Colorfield{B,T,M}, dU1::Colorfield{B,T,M}, U::Gaugefield{B,TU}, fac=1.0
) where {B,T,M,TU}
    @assert U.topology.numprocs_cart[4] == 1 """
    for polyakov loop, the field cannot be partitioned in the t-dimension
    """
    polyakov_deriv!(dU, U, 1, fac)
    polyakov_deriv!(dU1, U, -im, fac)
    _, _, _, trange = U.topology.bulk_sites.indices
    L = polyakov_traced(U)
    reL, imL, absL = real(L), imag(L), abs(L)

    parallelfor(eachindex(dU, U), B, Val(M), (), (U,), (dU, dU1, U)) do site, (dU, dU1, U)
        ix, iy, iz, it = site.I
        dU[4, site] = (reL/absL) * dU[4, site] + (imL/absL) * dU1[4, site]
    end

    return nothing
end

function polyakov_phase_deriv!(
    dU::Colorfield{B,T,M}, dU1::Colorfield{B,T,M}, U::Gaugefield{B,TU}, fac=1.0
) where {B,T,M,TU}
    @assert U.topology.numprocs_cart[4] == 1 """
    for polyakov loop, the field cannot be partitioned in the t-dimension
    """
    polyakov_deriv!(dU, U, 1, fac)
    polyakov_deriv!(dU1, U, -im, fac)
    _, _, _, trange = U.topology.bulk_sites.indices
    L = polyakov_traced(U)
    reL, imL, absL = real(L), imag(L), abs(L)

    parallelfor(eachindex(dU, U), B, Val(M), (), (U,), (dU, dU1, U)) do site, (dU, dU1, U)
        ix, iy, iz, it = site.I
        dU[4, site] = (-imL/absL^2) * dU[4, site] + (reL/absL^2) * dU1[4, site]
    end

    return nothing
end
