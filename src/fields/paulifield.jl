@field_constructor Paulifield extra_types=C extra_args=(csw, inverse)

# TODO: complete docs
@doc raw"""
Wrapper around a 4-dimensional dense array of `PauliMatrix`-objects contatining
information about the global MPI-topology.
A `PauliMatrix` is made up of 2 6x6 complex matrices, which are the upper left and lower
lower right blocks of a full 12x12 hermitian matrix.

    Paulifield{B,T,C}(NX, NY, NZ, NT, csw, inverse)
    Paulifield{B,T,C}(NX, NY, NZ, NT, csw, inverse; numprocs_cart, halo_width)
    Paulifield(f::AbstractField, csw, inverse; no_halo, hw)

Creates a Paulifield on `B`, i.e. an array of link-variables
of size `NX × NY × NZ × NT` or a zero-initialized copy of `f`.
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Paulifield

function Paulifield(
    u::AbstractField{B,T,M}, csw, inverse, ::Type{Tnew}=T; no_halo=false, hw=get_halo_width(u)
) where {B,T,M,Tnew}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Paulifield{B,Tnew}(
            size(u)..., csw, inverse;
            numprocs_cart=ncart, halo_width=hw, no_halo=no_halo
        )
    else
        Paulifield{B,Tnew}(size(u)..., csw, inverse)
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

function create_sendbuf!(p::Paulifield{B,T,M}, sites, dim, dir) where {B,T,M}
    ibuf = dir + 2(dim - 1)
    sendbuf = p.sendbuf[ibuf]
    itr = eachindex(IndexLinear(), sites)

    parallelfor(itr, B, Val(M), (), (), (p,)) do i, (p,)
        sendbuf[i] = p[sites[i]]
    end

    return mpi_make_transferrable(sendbuf)[1]
end

function Base.copyto!(a::TF, b::TF, arange, brange) where {B,T,M,TF<:Paulifield{B,T,M}}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    parallelfor(eachindex(IndexLinear(), arange), B, Val(M), (), (), (a, b)) do i, (a, b)
        site_a = arange[i]
        site_b = brange[i]
        a[site_a] = b[site_b]
    end

    return nothing
end
