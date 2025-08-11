abstract type AbstractFieldstrength end

struct Plaquette <: AbstractFieldstrength end
struct Clover <: AbstractFieldstrength end
struct Improved <: AbstractFieldstrength end

@field_constructor Tensorfield

@doc raw"""
Wrapper around a 6-dimensional dense array of statically sized 3x3 matrices contatining
information about the global MPI-topology.

    Tensorfield{B,T}(NX, NY, NZ, NT)
    Tensorfield{B,T}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Tensorfield(u::AbstractField)

Creates a `Tensorfield` on `B`, i.e. an array of 3-by-3 `T`-precision matrices
of size `6 × NX × NY × NZ × NT` or a zero-initialized Tensorfield of the same size as
`u`.
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Tensorfield

function Tensorfield(
    u::AbstractField{B,T,M}, ::Type{Tnew}=T; no_halo=false, hw=get_halo_width(u)
) where {B,T,M,Tnew}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Tensorfield{B,Tnew}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Tensorfield{B,Tnew}(size(u)...)
    end

    return u_out
end

const MPITensorfield{B,T} = Tensorfield{B,T,true}

Base.eltype(::Type{Tensorfield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}

### Indexing Start
# @inline allindices(u::Tensorfield{CPU}) = eachindex(IndexCartesian(), u.U) # all indices including halo regions
# @inline allindices(u::Tensorfield{B}) where {B} = 
#     CartesianIndices(ntuple(i -> ifelse(i == 5, axes(u.U, 6), axes(u.U, i)), Val(5)))
#     # CartesianIndices(ntuple(i -> axes(u.U, i), Val(5)))#eachindex(IndexCartesian(), u.U[1]) # all indices including halo regions
# Base.@propagate_inbounds Base.getindex(u::Tensorfield{CPU}, j, site::SiteCoords) = u.U[site, j]
# Base.@propagate_inbounds Base.getindex(u::Tensorfield{CPU}, jsite) = u.U[jsite]
#
# Base.@propagate_inbounds function Base.getindex(
#     f::Tensorfield{B,T}, j, site::SiteCoords
# ) where {B,T}
#     @inbounds begin
#         Base.Cartesian.@nexprs 9 i -> (
#             c_i = f.U[site, i, j]
#         )
#     end
#     return SMatrix{3,3,Complex{T},9}(c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9)
# end
#
# Base.@propagate_inbounds function Base.getindex(
#     f::Tensorfield{B,T}, jsite
# ) where {B,T}
#     @inbounds begin
#         x, y, z, t, j = jsite.I
#         Base.Cartesian.@nexprs 9 i -> (
#             c_i = f.U[x, y, z, t, i, j]
#         )
#     end
#     return SMatrix{3,3,Complex{T},9}(c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9)
# end
#
# Base.@propagate_inbounds Base.setindex!(f::Tensorfield{CPU}, v, j, site::SiteCoords) =
#     setindex!(f.U, v, site, j)
# Base.@propagate_inbounds Base.setindex!(f::Tensorfield{CPU}, v, jsite) =
#     setindex!(f.U, v, jsite)
#
# Base.@propagate_inbounds function Base.setindex!(
#     f::Tensorfield{B,T}, v, j, site::SiteCoords
# ) where {B,T}
#     @inbounds begin
#         Base.Cartesian.@nexprs 9 i -> (
#             f.U[site, i, j] = v[i];
#         )
#     end
#     return nothing
# end
#
# Base.@propagate_inbounds function Base.setindex!(
#     f::Tensorfield{B,T}, v, jsite
# ) where {B,T}
#     @inbounds begin
#         x, y, z, t, j = jsite.I
#         Base.Cartesian.@nexprs 9 i -> (
#             f.U[x, y, z, t, i, j] = v[i];
#         )
#     end
#     return nothing
# end

Base.@propagate_inbounds Base.getindex(u::Tensorfield, i, x, y, z, t) =
    u.U[x, y, z, t, i]
Base.@propagate_inbounds Base.getindex(u::Tensorfield, i, site::SiteCoords) =
    u.U[site, i]

Base.@propagate_inbounds function Base.getindex(u::MPITensorfield, i, site::SiteCoords)
    site in u.topology.bulk_sites && return u.U[site, i]
    ihalo = get_halo_index(site, u.topology.bulk_sites)
    return u.halos[ihalo][site, i]
end

Base.@propagate_inbounds Base.setindex!(u::Tensorfield, v, i, x, y, z, t) =
    setindex!(u.U, v, x, y, z, t, i)
Base.@propagate_inbounds Base.setindex!(u::Tensorfield, v, i, site::SiteCoords) =
    setindex!(u.U, v, site, i)
### Indexing End

Base.@propagate_inbounds function Base.setindex!(u::MPITensorfield, v, i, site::SiteCoords)
    bulk = u.topology.bulk_sites

    if site in bulk
        u.U[site, i] = v
    else
        ihalo = get_halo_index(site, bulk)
        u.halos[ihalo][site, i] = v
    end

    return nothing
end

@inline function get_tensor_index(μ, ν)
    lo, hi = minmax(μ, ν)
    return if lo == 1 && hi == 2
        1
    elseif lo == 1 && hi == 3
        2
    elseif lo == 1 && hi == 4
        3
    elseif lo == 2 && hi == 3
        4
    elseif lo == 2 && hi == 4
        5
    elseif lo == 3 && hi == 4
        6
    else
        throw(AssertionError("invalid tensor index combination"))
    end
end

function fieldstrength_eachsite!(F::Tensorfield, U, kind_of_fs::String)
    if kind_of_fs == "plaquette"
        fieldstrength_eachsite!(Plaquette(), F, U)
    elseif kind_of_fs == "clover"
        fieldstrength_eachsite!(Clover(), F, U)
    else
        error("kind of fieldstrength \"$(kind_of_fs)\" not supported")
    end

    return nothing
end

function fieldstrength_eachsite!(
    ::Plaquette, F::Tensorfield{B,T}, U::Gaugefield{B,T,M}
) where {B,T,M}
    fac = Complex{T}(im)

    parallelfor(eachindex(U, F), B, Val(M), (U,), (F,), (U, F)) do site, (U, F)
        C12 = plaquette(U, 1, 2, site)
        F[1, site] = fac * (C12 - C12')
        C13 = plaquette(U, 1, 3, site)
        F[2, site] = fac * (C13 - C13')
        C14 = plaquette(U, 1, 4, site)
        F[3, site] = fac * (C14 - C14')
        C23 = plaquette(U, 2, 3, site)
        F[4, site] = fac * (C23 - C23')
        C24 = plaquette(U, 2, 4, site)
        F[5, site] = fac * (C24 - C24')
        C34 = plaquette(U, 3, 4, site)
        F[6, site] = fac * (C34 - C34')
    end

    return nothing
end

function fieldstrength_eachsite!(
    ::Clover, F::Tensorfield{B,T}, U::Gaugefield{B,T,M}
) where {B,T,M}
    fac = Complex{T}(im / 8)

    parallelfor(eachindex(U, F), B, Val(M), (U,), (F,), (U, F)) do site, (U, F)
        C12 = clover_1x1(U, 1, 2, site)
        F[1, site] = fac * (C12 - C12')
        C13 = clover_1x1(U, 1, 3, site)
        F[2, site] = fac * (C13 - C13')
        C14 = clover_1x1(U, 1, 4, site)
        F[3, site] = fac * (C14 - C14')
        C23 = clover_1x1(U, 2, 3, site)
        F[4, site] = fac * (C23 - C23')
        C24 = clover_1x1(U, 2, 4, site)
        F[5, site] = fac * (C24 - C24')
        C34 = clover_1x1(U, 3, 4, site)
        F[6, site] = fac * (C34 - C34')
    end

    return nothing
end

function create_sendbuf!(F::Tensorfield{B,T,M}, sites, dim, dir) where {B,T,M}
    ibuf = dir + 2(dim - 1)
    sendbuf = F.sendbuf[ibuf]
    itr = eachindex(IndexLinear(), sites)

    parallelfor(itr, B, Val(M), (), (), (F,)) do i, (F,)
        site = sites[i]
        sendbuf[1, i] = F[1, site]
        sendbuf[2, i] = F[2, site]
        sendbuf[3, i] = F[3, site]
        sendbuf[4, i] = F[4, site]
        sendbuf[5, i] = F[5, site]
        sendbuf[6, i] = F[6, site]
    end

    return mpi_make_transferrable(sendbuf)[1]
end

function Base.copyto!(a::Tensorfield{B,T,M}, b::Tensorfield{B}, arange, brange) where {B,T,M}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    parallelfor(eachindex(IndexLinear(), arange), B, Val(M), (), (), (a, b)) do i, (a, b)
        site_a = arange[i]
        site_b = brange[i]
        a[1, site_a] = b[1, site_b]
        a[2, site_a] = b[2, site_b]
        a[3, site_a] = b[3, site_b]
        a[4, site_a] = b[4, site_b]
        a[5, site_a] = b[5, site_b]
        a[6, site_a] = b[6, site_b]
    end

    return nothing
end
