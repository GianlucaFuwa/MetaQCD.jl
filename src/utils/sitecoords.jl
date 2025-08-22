const SiteCoords = CartesianIndex{4}

"""
    move(s::SiteCoords, μ, steps, lim)

Move a site `s` in the direction `μ` by `steps` steps with periodic boundary conditions.
The maximum extent of the lattice in the direction `μ` is `lim`.
"""
@inline function move(s::SiteCoords, μ, steps, lim::Integer)
    return @set s[μ] = mod1(s[μ] + steps, lim)
end

@inline @inbounds function move(s::SiteCoords, μ, steps, r::AbstractUnitRange)
    iold = s[μ]
    len = length(r)
    offset = first(r)
    inew = offset + mod(iold + steps - offset, len)
    # @assert inew in r "$mpi_myrank(), $s, $μ, $steps, $r, $inew"
    return @set s[μ] = inew
end

@inline @inbounds function move(s::SiteCoords, ::Val{1}, steps, r::AbstractUnitRange)
    x, y, z, t = s.I
    len = Int32(length(r))
    offset = Int32(first(r))
    xnew = offset + mod(Int32(x) + Int32(steps) - offset, len)
    return SiteCoords(xnew, y, z, t)
end

@inline @inbounds function move(s::SiteCoords, ::Val{2}, steps, r::AbstractUnitRange)
    x, y, z, t = s.I
    len = Int32(length(r))
    offset = Int32(first(r))
    ynew = offset + mod(Int32(y) + Int32(steps) - offset, len)
    return SiteCoords(x, ynew, z, t)
end

@inline @inbounds function move(s::SiteCoords, ::Val{3}, steps, r::AbstractUnitRange)
    x, y, z, t = s.I
    len = Int32(length(r))
    offset = Int32(first(r))
    znew = offset + mod(Int32(z) + Int32(steps) - offset, len)
    return SiteCoords(x, y, znew, t)
end

@inline @inbounds function move(s::SiteCoords, ::Val{4}, steps, r::AbstractUnitRange)
    x, y, z, t = s.I
    len = Int32(length(r))
    offset = Int32(first(r))
    tnew = offset + mod(Int32(t) + Int32(steps) - offset, len)
    return SiteCoords(x, y, z, tnew)
end

Base.iseven(s::SiteCoords) = iseven(sum(s.I))
Base.isodd(s::SiteCoords) = isodd(sum(s.I))

@inline function map_to_half(site, irange) # XXX: Maybe write map_to_half(site, halo_range)
    site in irange || return site
    origin = first(irange)
    nx, ny, nz, _ = size(irange)
    nv = length(irange)
    i = cartesian_to_linear(site, nx, ny, nz, origin)
    offset = iseven(site) ? -fld(i, 2) : div(nv, 2) - fld(i, 2)
    i_new = i + offset
    return irange[i_new]
end

@inline map_to_half(site, irange, ::Nothing) = map_to_half(site, irange)

@inline function map_to_half(site, bulk, halo_sites)
    ihalo = get_halo_index(site, bulk)

    if ihalo == 0
        return map_to_half(site, bulk)
    else
        dim = cld(ihalo, 2)
        side = mod1(rem(ihalo, 2), 2)
        return map_to_half(site, halo_sites[dim][side])
    end
end

@inline function map_to_half_switch(site, irange)
    site in irange || return site
    origin = first(irange)
    nx, ny, nz, _ = size(irange)
    nv = length(irange)
    nvhalf = div(nv, 2)
    i = cartesian_to_linear(site, nx, ny, nz, origin)
    offset = iseven(site) ? -fld(i, 2) : nvhalf - fld(i, 2)
    i_new = i + offset
    i_new = i_new > nvhalf ? i_new - nvhalf : i_new + nvhalf
    return irange[i_new]
end

@inline function map_from_half(mapped_site, irange::CartesianIndices)
    origin = first(irange)
    nx, ny, nz, _ = size(irange)
    nv = length(irange)
    nvhalf = div(nv, 2)
    i_new = cartesian_to_linear(mapped_site, nx, ny, nz, origin)
    i = i_new <= nvhalf ? 2i_new : 2(i_new - nvhalf)

    site = linear_to_cartesian(i, nx, ny, nz, origin)
    if map_to_half(site, irange) == mapped_site
        return site
    else
        return linear_to_cartesian(i-1, nx, ny, nz, origin)
    end
end

"""
    switch_sides(site::CartesianIndex, NX, NY, NZ, NT, NV)

Return the cartesian index equivalent to `site` but with opposite parity.
E.g., `switch_sides((1, 1, 1, 1), 4, 4, 4, 4, 256) = (1, 1, 1, 3)` and reverse
"""
@inline function switch_sides(site::SiteCoords, irange)
    origin = first(irange)
    nx, ny, nz, _ = size(irange)
    nv = length(irange)
    i = cartesian_to_linear(site, nx, ny, nz, origin)
    nvhalf = div(nv, 2)
    i_new = i > nvhalf ? i - nvhalf : i + nvhalf
    return linear_to_cartesian(i_new, nx, ny, nz, origin)
end

@inline function switch_sides(
    site::SiteCoords, NX::T, NY::T, NZ::T, ::T, NV::T
) where {T<:Integer}
    i = cartesian_to_linear(site, NX, NY, NZ)
    nvhalf = div(NV, 2)
    i_new = i > nvhalf ? i - nvhalf : i + nvhalf
    return linear_to_cartesian(i_new, NX, NY, NZ)
end

@inline function get_halo_index(site, bulk)
    site in bulk && return 0
    
    # Extract coordinates and bulk bounds
    x, y, z, t = Tuple(site)
    bulk_start = Tuple(first(bulk))
    bulk_end = Tuple(last(bulk))
    x_min, y_min, z_min, t_min = bulk_start
    x_max, y_max, z_max, t_max = bulk_end
    
    # Determine which halo region this site belongs to
    # Halo numbering: 1=-x, 2=+x, 3=-y, 4=+y, 5=-z, 6=+z, 7=-t, 8=+t
    
    # Check t direction first (highest priority)
    if t > t_max
        halo_id = 8
    elseif t < t_min
        halo_id = 7
    elseif z > z_max
        halo_id = 6
    elseif z < z_min
        halo_id = 5
    elseif y > y_max
        halo_id = 4
    elseif y < y_min
        halo_id = 3
    elseif x > x_max
        halo_id = 2
    elseif x < x_min
        halo_id = 1
    end
    # if x < x_min
    #     halo_id = 1
    # elseif x > x_max
    #     halo_id = 2
    # elseif y < y_min
    #     halo_id = 3
    # elseif y > y_max
    #     halo_id = 4
    # elseif z < z_min
    #     halo_id = 5
    # elseif z > z_max
    #     halo_id = 6
    # elseif t < t_min
    #     halo_id = 7
    # elseif t > t_max
    #     halo_id = 8
    # end
    
    return halo_id
end

@inline function linear_to_cartesian(i::T, NX::T, NY::T, NZ::T) where {T<:Integer}
    ix = (i - 1) % NX + 1
    ii = div(i - ix, NX)

    iy = ii % NY + 1
    ii = div(ii - (iy - 1), NY)

    iz = ii % NZ + 1
    it = div(ii - (iz - 1), NZ) + 1
    return SiteCoords(ix, iy, iz, it)
end

@inline function linear_to_cartesian(
    i::T, NX::T, NY::T, NZ::T, origin::SiteCoords
) where {T<:Integer}
    ix = (i - 1) % NX + 1
    ii = div(i - ix, NX)

    iy = ii % NY + 1
    ii = div(ii - (iy - 1), NY)

    iz = ii % NZ + 1
    it = div(ii - (iz - 1), NZ) + 1
    return SiteCoords(ix, iy, iz, it) - (CartesianIndex(1, 1, 1, 1) - origin)
end

@inline function cartesian_to_linear(
    site::SiteCoords, NX::T, NY::T, NZ::T
) where {T<:Integer}
    ix, iy, iz, it = site.I
    i = ix + NX * (iy - 1 + NY * (iz - 1 + NZ * (it - 1)))
    return i
end

@inline function cartesian_to_linear(
    offsite::SiteCoords, NX::T, NY::T, NZ::T, origin::SiteCoords
) where {T<:Integer}
    site = offsite + (CartesianIndex(1, 1, 1, 1) - origin) 
    ix, iy, iz, it = site.I
    i = ix + NX * (iy - 1 + NY * (iz - 1 + NZ * (it - 1)))
    return i
end

@inline function halo_to_full(hsite, global_dims)
    NX, NY, NZ, NT = global_dims
    ix, iy, iz, it = hsite.I
    x = mod1(ix, NX)
    y = mod1(iy, NY)
    z = mod1(iz, NZ)
    t = mod1(it, NT)
    return SiteCoords(x, y, z, t)
end
