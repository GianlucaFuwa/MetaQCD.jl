struct Paulifield{B,T,M,C,AT,TT} <: AbstractField{B,T,M,AT}
    U::AT
    halos::Nothing # XXX: ugly
    sendbuf::Nothing
    topology::TT
    csw::Float64
    function Paulifield{B,T,M,C}(
        U::AT, halos, sendbuf, topology::TT, csw, ::Bool
    ) where {B,T,M,C,AT,TT}
        check_types(B, T, U, halos, sendbuf)
        return new{B,T,M,C,AT,TT}(U, nothing, nothing, topology, csw)
    end
end

@field_constructor Paulifield extra_types=C extra_args=(csw, inverse)

function Paulifield(
    u::AbstractField{B,T,M}, csw, inverse; no_halo=false, hw=get_halo_width(u)
) where {B,T,M}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Paulifield{B,T}(
            size(u)..., csw, inverse;
            numprocs_cart=ncart, halo_width=hw, no_halo=no_halo
        )
    else
        Paulifield{B,T}(size(u)..., csw, inverse)
    end

    return u_out
end

const MPIPaulifield{B,T} = Paulifield{B,T,true}

Base.eltype(::Type{Paulifield}, ::Type{T}) where {T} = PauliMatrix{6,36,T}
@inline has_clover_term(::Paulifield{B,T,M,C}) where {B,T,M,C} = C

Base.@propagate_inbounds Base.getindex(p::Paulifield, i::Integer) = p.U[i]
Base.@propagate_inbounds Base.getindex(p::Paulifield, x, y, z, t) = p.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(p::Paulifield, site::SiteCoords) = p.U[site]

Base.@propagate_inbounds function Base.getindex(u::MPIPaulifield, site::SiteCoords)
    site in u.topology.bulk_sites && return u.U[site]
    ihalo = get_halo_index(site, u.topology.bulk_sites)
    return u.halos[ihalo][site]
end

Base.@propagate_inbounds Base.setindex!(p::Paulifield, v, i::Integer) =
    setindex!(p.U, v, i)
Base.@propagate_inbounds Base.setindex!(p::Paulifield, v, x, y, z, t) =
    setindex!(p.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(p::Paulifield, v, site::SiteCoords) =
    setindex!(p.U, v, site)

Base.@propagate_inbounds function Base.setindex!(u::MPIPaulifield, v, site::SiteCoords)
    bulk = u.topology.bulk_sites

    if site in bulk
        u.U[site] = v
    else
        ihalo = get_halo_index(site, bulk)
        u.halos[ihalo][site] = v
    end

    return nothing
end

function create_sendbuf!(p::Paulifield, sites, dim, dir)
    ibuf = dir + 2(dim - 1)
    sendbuf = p.sendbuf[ibuf]

    @batch for i in eachindex(IndexLinear(), sites)
        sendbuf[i] = p[sites[i]]
    end

    return sendbuf
end

function Base.copyto!(a::Paulifield, b::Paulifield, arange, brange)
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    @batch for i in eachindex(IndexLinear(), arange)
        site_a = arange[i]
        site_b = brange[i]
        a[site_a] = b[site_b]
    end

    return nothing
end
