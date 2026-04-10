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
    u::AbstractField{B,T,M}, ::Type{Tnew}=T; no_halo=false, halo_width=get_halo_width(u)
) where {B,T,M,Tnew}
    u_out = if M
        numprocs_cart = get_numprocs_cart(u)
        Colorfield{B,Tnew}(size(u)...; numprocs_cart, halo_width, no_halo)
    else
        Colorfield{B,Tnew}(size(u)...)
    end

    return u_out
end

Base.eltype(::Type{Colorfield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}

#### CPU Indexing ####
Base.@propagate_inbounds Base.getindex(u::Colorfield{CPU}, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.getindex(u::Colorfield{CPU}, μsite) = u.U[μsite]
Base.@propagate_inbounds Base.setindex!(u::Colorfield{CPU}, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)
Base.@propagate_inbounds Base.setindex!(u::Colorfield{CPU}, v, μsite) =
    setindex!(u.U, v, μsite)
######################

#### GPU Indexing ####
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

Base.@propagate_inbounds function Base.getindex(
    u::Colorfield{B,T}, μsite
) where {B,T}
    return _getindex_mat(Val(18), u.U, μsite, T)
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

Base.@propagate_inbounds function Base.setindex!(
    u::Colorfield{B,T}, v, μsite
) where {B,T}
    return _setindex_mat!(Val(18), u.U, v, μsite, T)
end

function convert_field(
    ::Type{Bout}, Uin::Colorfield{CPU,Tin,M}, ::Type{Tout}=Tin
) where {M,Bout,Tout,Tin}
    if Bout === CPU
        Uout = similar(Uin, Tout)
        copy!(Uout, Uin)
        return Uout
    end

    NX, NY, NZ, NT = size(Uin)
    numprocs_cart = get_numprocs_cart(Uin)
    halo_width = get_halo_width(Uin)
    Uout = Colorfield{Bout,Tout}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Uarr = OffsetArray(array_type(Bout)(Uin.U.parent), OffsetArrays.Origin(Uin.U))

    parallelfor(eachindex(Uout), Bout, Val(M), (Uout,), (), (Uout,)) do site, (Uout,)
        Uout[1, site] = Uarr[1, site]
        Uout[2, site] = Uarr[2, site] 
        Uout[3, site] = Uarr[3, site] 
        Uout[4, site] = Uarr[4, site] 
    end

    return Uout
end

function convert_field(
    ::Type{Bout}, Uin::Colorfield{Bin,Tin,M}, ::Type{Tout}=Tin
) where {M,Bout,Tout,Bin,Tin}
    if Bout === Bin
        Uout = similar(Uin, Tout)
        copy!(Uout, Uin)
        return Uout
    end

    NX, NY, NZ, NT = size(Uin)
    numprocs_cart = get_numprocs_cart(Uin)
    halo_width = get_halo_width(Uin)
    Uout = Colorfield{Bout,Tout}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Uarr = OffsetArray(array_type(Bout)(Uin.U.parent), OffsetArrays.Origin(Uin.U))

    parallelfor(eachindex(Uout), Bout, Val(M), (Uout,), (), (Uout,)) do site, (Uout,)
        Uout[1, site] = _getindex_mat(Val(18), Uarr, 1, site, Tout)
        Uout[2, site] = _getindex_mat(Val(18), Uarr, 2, site, Tout)
        Uout[3, site] = _getindex_mat(Val(18), Uarr, 3, site, Tout)
        Uout[4, site] = _getindex_mat(Val(18), Uarr, 4, site, Tout)
    end

    return Uout
end
