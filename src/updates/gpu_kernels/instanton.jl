# FIXME:
function set_instanton!(U::Gaugefield{B,T}, Q) where {B<:GPU,T}
    NX, NY, NZ, NT = size(U)
    nx, ny, nz, nt = get_local_dims(U)
    bulk_sites = U.topology.bulk_sites
    workgroupsize = (4, 4, 4, 4)

    s = sig(T)
    s_comp = sig_comp(T)
    s_id = sig_id(T)

    field_x = T(2π * abs(Q) / NX)
    field_t = T(2π * abs(Q) / (NX*NT))

    transform_x! = instanton_x_kernel!(B(), workgroupsize)
    transform_x!(U.U, field_t, s, s_id, s_comp, bulk_sites; ndrange=(nx, ny, nz, nt))
    synchronize(B())

    idc = SVector{3,Int64}(1, 2, 3)
    bulk_sites_t = ntuple(i -> bulk_sites.indices[idc[i]], Val(3))
    transform_t! = instanton_t_kernel!(B(), workgroupsize)
    transform_t!(U.U, field_t, s, s_id, s_comp, NT, bulk_sites_t; ndrange=(nx, ny, nz))
    synchronize(B())

    if Q == 0
        field_y = T(0)
        field_z = T(0)
    else
        field_y = T(-2π * Q / (abs(Q) * NY * NZ))
        field_z = T(-2π * Q / (abs(Q) * NZ))
    end

    transform_z! = instanton_z_kernel!(B(), workgroupsize)
    transform_z!(U.U, field_y, t, t_id, t_comp, bulk_sites; ndrange=(nx, ny, nz, nt))
    synchronize(B())

    idc = SVector{3,Int64}(1, 3, 4)
    bulk_sites_y = ntuple(i -> bulk_sites.indices[idc[i]], Val(3))
    transform_y! = instanton_y_kernel!(B(), workgroupsize)
    transform_y!(U.U, field_z, t, t_id, t_comp, NY, bulk_sites_y; ndrange=(nx, nz, nt))
    synchronize(B())
    return nothing
end

@kernel function instanton_x_kernel!(U, field_t, s, s_id, s_comp, bulk_sites)
    ii = @index(Global, Linear)
    site = bulk_sites[ii]
    it = site[4]
    cit = cos(field_t * it)
    sit = sin(field_t * it)
    @inbounds U[1, site] = s_comp + cit * s_id + im * sit * s
end

@kernel function instanton_y_kernel!(U, field_z, t, t_id, t_comp, NY, bulk_sites_y)
    ii = @index(Global, Linear)
    site = bulk_sites_y[ii]
    ix, iz, it = site[1], site[3], site[4]
    cit = cos(field_z * iz)
    sit = sin(field_z * iz)
    @inbounds U[2, ix, NY, iz, it] = t_comp + cit * t_id - im * sit * t
end

@kernel function instanton_z_kernel!(U, field_y, t, t_id, t_comp, bulk_sites)
    ii = @index(Global, Linear)
    site = bulk_sites[ii]
    iy = site[2]
    cit = cos(field_y * iy)
    sit = sin(field_y * iy)
    @inbounds U[3, site] = t_comp + cit * t_id + im * sit * t
end

@kernel function instanton_t_kernel!(U, field_x, s, s_id, s_comp, NT, bulk_sites_t)
    ii = @index(Global, Linear)
    site = bulk_sites_t[ii]
    ix, iy, iz = site[1], site[2], site[3]
    cit = cos(field_x * ix)
    sit = sin(field_x * ix)
    @inbounds U[4, ix, iy, iz, NT] = s_comp + cit * s_id - im * sit * s
end
