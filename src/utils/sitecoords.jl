const SiteCoords = CartesianIndex{4}

@inline function linear_to_cartesian(i::T, NX::T, NY::T, NZ::T) where {T<:Integer}
    ix = (i - 1) % NX + 1
    ii = div(i - ix, NX)

    iy = ii % NY + 1
    ii = div(ii - (iy - 1), NY)

    iz = ii % NZ + 1
    it = div(ii - (iz - 1), NZ) + 1
    return SiteCoords(ix, iy, iz, it)
end

@inline function cartesian_to_linear(
    site::SiteCoords, NX::T, NY::T, NZ::T
) where {T<:Integer}
    ix, iy, iz, it = site.I
    i = ix + NX * (iy - 1) + NX * NY * (iz - 1) + NX * NY * NZ * (it - 1)
    return i
end

"""
    switch_sides(site::CartesianIndex, NX, NY, NZ, NT, NV)

Return the cartesian index equivalent to `site` but with opposite parity.
E.g., `switch_sides((1, 1, 1, 1), 4, 4, 4, 4, 256) = (1, 1, 1, 3)` and reverse
"""
@inline function switch_sides(
    site::SiteCoords, NX::T, NY::T, NZ::T, ::T, NV::T
) where {T<:Integer}
    i = cartesian_to_linear(site, NX, NY, NZ)
    nvhalf = div(NV, 2)
    i_new = i > nvhalf ? i - nvhalf : i + nvhalf
    return linear_to_cartesian(i_new, NX, NY, NZ)
end

"""
    move(s::SiteCoords, μ, steps, lim)

Move a site `s` in the direction `μ` by `steps` steps with periodic boundary conditions.
The maximum extent of the lattice in the direction `μ` is `lim`.
"""
@inline move(s::SiteCoords, μ, steps, lim) = @set s[μ] = mod1(s[μ] + steps, lim)
Base.iseven(s::SiteCoords) = iseven(sum(s.I))
Base.isodd(s::SiteCoords) = isodd(sum(s.I))

@inline function eo_site(site, NX::T, NY::T, NZ::T, ::T, NV::T) where {T<:Integer}
    i = cartesian_to_linear(site, NX, NY, NZ)
    offset = iseven(site) ? -fld(i, 2) : div(NV, 2) - fld(i, 2)
    i_new = i + offset
    return linear_to_cartesian(i_new, NX, NY, NZ)
end

@inline function eo_site_switch(site, NX::T, NY::T, NZ::T, ::T, NV::T) where {T<:Integer}
    nvhalf = div(NV, 2)
    i = cartesian_to_linear(site, NX, NY, NZ)
    offset = iseven(site) ? -fld(i, 2) : nvhalf - fld(i, 2)
    i_new = i + offset
    i_new = i_new > nvhalf ? i_new - nvhalf : i_new + nvhalf
    return linear_to_cartesian(i_new, NX, NY, NZ)
end
