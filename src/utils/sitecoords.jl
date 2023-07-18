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

@inline function move(s::SiteCoords, μ, steps, lim)
    ix, iy, iz, it = Tuple(s)

    if μ == 1
        ix = mod1(ix + steps, lim)
    elseif μ == 2
        iy = mod1(iy + steps, lim)
    elseif μ == 3
        iz = mod1(iz + steps, lim)
    elseif μ == 4
        it = mod1(it + steps, lim)
    end

    return SiteCoords(ix, iy, iz, it)
end
