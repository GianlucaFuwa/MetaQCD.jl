struct SiteCoords
    ix::Int64
    iy::Int64
    iz::Int64
    it::Int64
end

@inline function SiteCoords(i, NX, NY, NZ, NT)
    ix = (i - 1) % NX + 1
    ii = div(i - ix, NX)

    iy = ii % NY + 1
    ii = div(ii - (iy - 1), NY)

    iz = ii % NZ + 1
    it = div(ii - (iz - 1), NZ) + 1
    return SiteCoords(ix, iy, iz, it)
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

function Base.setindex!(
    U::Array{SMatrix{3, 3, ComplexF64, 9}, 4},
    v::SMatrix{3, 3, ComplexF64, 9},
    s::SiteCoords,
)
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