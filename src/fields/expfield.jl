@field_constructor Expfield

"""
Wrapper around a 5-dimensional dense array of `ExpiQCoeffs` objects contatining
information about the global MPI-topology. The array holds the `Q`-matrices and all the
exponential parameters needed for stout-force recursion.

    Expfield{B,T}(NX, NY, NZ, NT)
    Expfield{B,T}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Expfield(u::AbstractField)

Creates a Expfield on `B`, i.e. an array of `T`-precison `ExpiQCoeffs` objects
of size `4 × NX × NY × NZ × NT` or of the same size as `u`.
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Expfield

function Expfield(
    u::AbstractField{B,T,M}, ::Type{Tnew}=T; no_halo=false, halo_width=get_halo_width(u)
) where {B,T,M,Tnew}
    u_out = if M
        numprocs_cart = get_numprocs_cart(u)
        Expfield{B,Tnew}(size(u)...; numprocs_cart, halo_width, no_halo)
    else
        Expfield{B,Tnew}(size(u)...)
    end

    return u_out
end

Base.eltype(::Type{Expfield}, ::Type{T}) where {T} = ExpiQCoeffs{T}

#### CPU Indexing ####
Base.@propagate_inbounds Base.getindex(u::Expfield{CPU}, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.getindex(u::Expfield{CPU}, μsite) = u.U[μsite]
Base.@propagate_inbounds Base.setindex!(u::Expfield{CPU}, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)
Base.@propagate_inbounds Base.setindex!(u::Expfield{CPU}, v, μsite) =
    setindex!(u.U, v, μsite)
######################

#### GPU Indexing ####
Base.@propagate_inbounds function Base.getindex(
    u::Expfield{B,T}, ii::Integer
) where {B,T}
    return u.U[ii]
end

Base.@propagate_inbounds function Base.getindex(
    u::Expfield{B,T}, μ, site::SiteCoords
) where {B,T}
    return u.U[site, μ]
end

Base.@propagate_inbounds function Base.setindex!(u::Expfield{B}, v, ii::Integer) where {B}
    u.U[ii] = v
    return nothing
end

Base.@propagate_inbounds function Base.setindex!(
    u::Expfield{B,T}, v, μ, site::SiteCoords
) where {B,T}
    u.U[site, μ] = v
    return nothing
end
