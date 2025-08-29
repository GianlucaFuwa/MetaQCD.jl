@field_constructor Colorfield

@doc raw"""
Wrapper around a 5-dimensional dense array of statically sized 3x3 matrices contatining
information about the global MPI-topology.

    Colorfield{B,T}(NX, NY, NZ, NT)
    Colorfield{B,T}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Colorfield(u::AbstractField)

Creates a Colorfield on `B`, i.e. an array of 3-by-3 `T`-precision matrices of
size `4 × NX × NY × NZ × NT` or a zero-initialized Colorfield of the same size as `u`
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Colorfield

function Colorfield(
    u::AbstractField{B,T,M}, ::Type{Tnew}=T; no_halo=false, hw=get_halo_width(u)
) where {B,T,M,Tnew}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Colorfield{B,Tnew}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Colorfield{B,Tnew}(size(u)...)
    end

    return u_out
end

Base.eltype(::Type{Colorfield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}

#### CPU Indexing ####
@inline allindices(u::Colorfield{CPU}) = eachindex(IndexCartesian(), u.U)
Base.@propagate_inbounds Base.getindex(u::Colorfield{CPU}, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.getindex(u::Colorfield{CPU}, μsite) = u.U[μsite]
Base.@propagate_inbounds Base.setindex!(u::Colorfield{CPU}, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)
Base.@propagate_inbounds Base.setindex!(u::Colorfield{CPU}, v, μsite) =
    setindex!(u.U, v, μsite)
######################

#### GPU Indexing ####
@inline allindices(u::Colorfield{B}) where {B} = 
    range(Int32(1), Int32(length(u.U)))

Base.@propagate_inbounds function Base.getindex(
    u::Colorfield{B,T}, ii::Integer
) where {B,T}
    return u.U[ii]
end

Base.@propagate_inbounds function Base.getindex(
    u::Colorfield{B,T}, μ, site::SiteCoords
) where {B,T}
    return _getindex_mat(Val(18), u.U, μ, site, T)
end

Base.@propagate_inbounds function Base.setindex!(u::Colorfield{B}, v, ii::Integer) where {B}
    u.U[ii] = v
    return nothing
end

Base.@propagate_inbounds function Base.setindex!(
    u::Colorfield{B,T}, v, μ, site::SiteCoords
) where {B,T}
    return _setindex_mat!(Val(18), u.U, v, μ, site, T)
end
######################
