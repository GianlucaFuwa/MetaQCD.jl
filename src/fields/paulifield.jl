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

Base.eltype(::Type{Paulifield}, ::Type{T}) where {T} = PauliMatrix{6,36,T}
@inline has_clover_term(::Paulifield{B,T,M,C}) where {B,T,M,C} = C

Base.@propagate_inbounds Base.getindex(p::Paulifield, i::Integer) = p.U[i]
Base.@propagate_inbounds Base.getindex(p::Paulifield, site::SiteCoords) = p.U[site]
Base.@propagate_inbounds Base.setindex!(p::Paulifield, v, i::Integer) =
    setindex!(p.U, v, i)
Base.@propagate_inbounds Base.setindex!(p::Paulifield, v, site::SiteCoords) =
    setindex!(p.U, v, site)

@inline allindices(p::Paulifield) = eachindex(p.U)

# #### CPU Indexing ####
# Base.@propagate_inbounds Base.getindex(u::Paulifield{CPU}, μ, site::SiteCoords) = u.U[μ, site]
# Base.@propagate_inbounds Base.getindex(u::Paulifield{CPU}, μsite) = u.U[μsite]
# Base.@propagate_inbounds Base.setindex!(u::Paulifield{CPU}, v, μ, site::SiteCoords) =
#     setindex!(u.U, v, μ, site)
# Base.@propagate_inbounds Base.setindex!(u::Paulifield{CPU}, v, μsite) =
#     setindex!(u.U, v, μsite)
# ######################
#
# #### GPU Indexing ####
# Base.@propagate_inbounds function Base.getindex(
#     u::Paulifield{B,T}, ii::Integer
# ) where {B,T}
#     return u.U[ii]
# end
#
# Base.@propagate_inbounds function Base.getindex(u::Paulifield{B,T}, site) where {B,T}
#     return _getindex_mat(Val(72), u.U, site, T)
# end
#
# Base.@propagate_inbounds function Base.setindex!(u::Paulifield{B}, v, ii::Integer) where {B}
#     u.U[ii] = v
#     return nothing
# end
#
# Base.@propagate_inbounds function Base.setindex!(u::Paulifield{B,T}, v, site) where {B,T}
#     return _setindex_mat!(Val(72), u.U, v, site, T)
# end
#
# Base.@propagate_inbounds function Base.getindex(u::MPIColorfield{B,T}, site) where {B,T}
#     site in u.topology.bulk_sites && return _getindex_mat(Val(72), u.U, site, T)
#     ihalo = get_halo_index(site, u.topology.bulk_sites)
#     return _getindex_mat(Val(72), u.halos[ihalo], site, T)
# end
#
# Base.@propagate_inbounds function Base.setindex!(u::MPIColorfield{B,T}, v, site) where {B,T}
#     bulk = u.topology.bulk_sites
#
#     if site in bulk
#         _setindex_mat!(Val(72), u.U, v, site, T)
#     else
#         ihalo = get_halo_index(site, bulk)
#         _setindex_mat!(Val(72), u.halos[ihalo], v, site, T)
#     end
#
#     return nothing
# end
#
# Base.@propagate_inbounds function _getindex_mat(
#     ::Val{72}, arr, site, ::Type{T}
# ) where {T}
#     x, y, z, t = site.I
#     Base.Cartesian.@nexprs 18 i -> (
#         vec = arr[x, y, z, t, i];
#         c_{2(i-1)+1} = Complex(vec[1], vec[2]);
#         c_{2(i-1)+2} = Complex(vec[3], vec[4]);
#     )
#     return SMatrix{6,6,Complex{T},36}(Base.Cartesian.@ntuple 18 c)
# end
#
# Base.@propagate_inbounds function _setindex_mat!(
#     ::Val{72}, arr, v, site, ::Type{T}
# ) where {T}
#     x, y, z, t = site.I
#     Base.Cartesian.@nexprs 18 i -> (
#         v1 = v[2(i-1)+1];
#         v2 = v[2(i-1)+2];
#         arr[x, y, z, t, i] = SIMD.Vec{4,T}((v1.re, v1.im, v2.re, v2.im))
#     )
#     return nothing
# end
# ######################

function create_sendbuf!(p::Paulifield{B,T,M}, sites, dim, dir) where {B,T,M}
    ibuf = dir + 2(dim - 1)
    sendbuf = p.sendbuf[ibuf]
    itr = eachindex(IndexLinear(), sites)

    parallelfor(itr, B, Val(M), (), (), (p, sendbuf)) do i, (p, sendbuf)
        sendbuf[i] = p[sites[i]]
    end

    return mpi_make_transferrable(sendbuf)
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

function fill_halo!(p::Paulifield{B,T,M}, recvbuf, siterange) where {B,T,M}
    itr = eachindex(IndexLinear(), siterange)

    parallelfor(itr, B, Val(M), (), (), (p, recvbuf)) do i, (p, recvbuf)
        site = siterange[i]
        p[site] = recvbuf[i]
    end

    return nothing
end
