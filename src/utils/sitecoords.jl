const SiteCoords = CartesianIndex{4}

@inline function linear_coords(i, NX, NY, NZ)
    ix = (i - 1) % NX + 1
    ii = div(i - ix, NX)

    iy = ii % NY + 1
    ii = div(ii - (iy - 1), NY)

    iz = ii % NZ + 1
    it = div(ii - (iz - 1), NZ) + 1
    return SiteCoords(ix, iy, iz, it)
end

"""
    move(s::SiteCoords, μ, steps, lim)

Move a site `s` in the direction `μ` by `steps` steps with periodic boundary conditions.
The maximum extent of the lattice in the direction `μ` is `lim`.
"""
move(s::SiteCoords, μ, steps, lim) = @set s[μ] = mod1(s[μ] + steps, lim)
