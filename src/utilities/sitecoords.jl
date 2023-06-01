struct SiteCoords{T}
    ix::T
    iy::T
    iz::T
    it::T
end 

@inline function get_coords(s::SiteCoords)
    return (s.ix, s.iy, s.iz, s.it)
end

@inline function move(s::SiteCoords, μ, steps, lim)
    ix, iy, iz, it = get_coords(s)

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

function Base.setindex!(U::AbstractArray, v, s::SiteCoords)
    ix, iy, iz, it = get_coords(s)
    @inbounds U[ix,iy,iz,it] = v
    return nothing
end

@inline function Base.getindex(U::AbstractArray, s::SiteCoords)
    ix, iy, iz, it = get_coords(s)
    @inbounds return U[ix,iy,iz,it] 
end

function Base.to_indices(U, s::SiteCoords)
    return to_indices(U, get_coords(s))
end

function Base.to_index(U, s::SiteCoords)
    return Base.to_indices(U, s)
end