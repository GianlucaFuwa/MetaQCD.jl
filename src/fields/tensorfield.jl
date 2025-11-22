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

Base.eltype(::Type{Tensorfield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}

#### CPU Indexing ####
@inline function add_directional_indices(::Tensorfield{CPU}, siterange::CartesianIndices)
    return CartesianIndices((6, siterange.indices...))
end

@inline function add_all_indices(::Tensorfield{CPU}, siterange::CartesianIndices)
    return CartesianIndices((6, siterange.indices...))
end

Base.@propagate_inbounds Base.getindex(u::Tensorfield{CPU}, i, site::SiteCoords) = u.U[i, site]
Base.@propagate_inbounds Base.getindex(u::Tensorfield{CPU}, isite) = u.U[isite]
Base.@propagate_inbounds Base.setindex!(u::Tensorfield{CPU}, v, i, site::SiteCoords) =
    setindex!(u.U, v, i, site)
Base.@propagate_inbounds Base.setindex!(u::Tensorfield{CPU}, v, isite) =
    setindex!(u.U, v, isite)
######################

#### GPU Indexing ####
@inline function add_directional_indices(::Tensorfield{B}, siterange::CartesianIndices) where {B}
    return CartesianIndices((siterange.indices..., 6))
end

@inline function add_all_indices(::Tensorfield{B}, siterange::CartesianIndices) where {B}
    return CartesianIndices((9, siterange.indices..., 6))
end

Base.@propagate_inbounds function Base.getindex(
    u::Tensorfield{B,T}, ii::Integer
) where {B,T}
    return u.U[ii]
end

Base.@propagate_inbounds function Base.getindex(
    u::Tensorfield{B,T}, i, site::SiteCoords
) where {B,T}
    return _getindex_mat(Val(18), u.U, i, site, T)
end

Base.@propagate_inbounds function Base.getindex(
    u::Tensorfield{B,T}, isite
) where {B,T}
    return _getindex_mat(Val(18), u.U, isite, T)
end

Base.@propagate_inbounds function Base.setindex!(u::Tensorfield{B}, v, ii::Integer) where {B}
    u.U[ii] = v
    return nothing
end

Base.@propagate_inbounds function Base.setindex!(
    u::Tensorfield{B,T}, v, i, site::SiteCoords
) where {B,T}
    return _setindex_mat!(Val(18), u.U, v, i, site, T)
end

Base.@propagate_inbounds function Base.setindex!(
    u::Tensorfield{B,T}, v, isite
) where {B,T}
    return _setindex_mat!(Val(18), u.U, v, isite, T)
end
######################

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
    ::Plaquette, F::Tensorfield{B,T}, U::Gaugefield{B,TU,M}
) where {B,T,M,TU}
    fac = Complex{T}(im)

    parallelfor(eachindex(U, F), B, Val(M), (U,), (F,), (U, F); do_edges=Val(true)) do site, (U, F)
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
    ::Clover, F::Tensorfield{B,T}, U::Gaugefield{B,TU,M}
) where {B,T,M,TU}
    fac = Complex{T}(im / 8)

    parallelfor(eachindex(U, F), B, Val(M), (U,), (F,), (U, F); do_edges=Val(true)) do site, (U, F)
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
