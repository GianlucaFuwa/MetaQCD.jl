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
#
# struct EvenOddFilter{I} <: AbstractArray{CartesianIndex{4},4}
#     pred::Bool
#     L::Int
#     itr::I
# end
#
# """
#     eo(even::Bool, itr::CartesianIndices)
#
# Create an iterable object that can be used in a for loop to iterate over all sites
# with parity `even`
# """
# function eo(even::Bool, itr)
#     @assert iseven(length(itr))
#     return EvenOddFilter(even, div(length(itr), 2), itr)
# end
#
# Base.length(eo::EvenOddFilter) = div(length(eo.itr), 2)
# Base.size(eo::EvenOddFilter) = size(eo.itr)
#
# @inline function Base.getindex(eo::EvenOddFilter{R}, i::Int) where {R}
#     # offset = if isodd(i) == eo.pred
#     #     iseven(div(i, 4)) ?  
#     # i += offset
#     i += iseven(i) == eo.pred
#     while true
#         iseven(eo.itr[i]) == eo.pred && return eo.itr[i]
#         i += 1
#     end
# end
#
# @inline function Base.getindex(eo::EvenOddFilter{R}, site::SiteCoords) where {R}
#     while true
#         iseven(eo.itr[site]) == eo.pred && return eo.itr[site]
#         valid, I = Base.IteratorsMD.__inc(site.I, eo.itr.indices)
#         valid || return nothing
#         site = CartesianIndex(I...)
#     end
# end
#
# @inline function Base.first(eo::EvenOddFilter)
#     iterfirst = eo.pred ? CartesianIndex(1, 1, 1, 1) : CartesianIndex(2, 1, 1, 1)
#     return iterfirst
# end
#
# @inline function Base.iterate(eo::EvenOddFilter)
#     iterfirst = first(eo)
#     return iterfirst, iterfirst
# end
#
# @inline function Base.iterate(eo::EvenOddFilter, state::CartesianIndex{4})
#     valid, I = Base.IteratorsMD.__inc(state.I, eo.itr.indices)
#     valid || return nothing
#     if iseven(sum(I)) == eo.pred
#         return CartesianIndex(I...), CartesianIndex(I...)
#     else
#         return iterate(eo, CartesianIndex(I...))
#     end
# end
#
# @inline function Base.iterate(eo::EvenOddFilter, state::Int)
#     valid = state <= eo.L
#     I = linear_to_cartesian(state + 1, map(x -> x.stop, Base.front(eo.itr.indices))...).I
#     valid || return nothing
#     if iseven(sum(I)) == eo.pred
#         return state + 1, state + 1
#     else
#         return iterate(eo, state + 1)
#     end
# end
