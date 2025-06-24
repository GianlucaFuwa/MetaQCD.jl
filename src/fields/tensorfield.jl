abstract type AbstractFieldstrength end

struct Plaquette <: AbstractFieldstrength end
struct Clover <: AbstractFieldstrength end
struct Improved <: AbstractFieldstrength end

"""
6-dimensional dense array of statically sized 3x3 matrices contatining associated meta-data.

    Tensorfield{B,T}(NX, NY, NZ, NT)
    Tensorfield{B,T}(NX, NY, NZ, NT, numprocs_cart, halo_width)
    Tensorfield(u::AbstractField)
    Tensorfield(parameters::ParameterSet)

Creates a `Tensorfield` on `B`, i.e. an array of 3-by-3 `T`-precision matrices
of size `4 x 4 × NX × NY × NZ × NT` or a zero-initialized Tensorfield of the same size as
`u`.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Tensorfield{B,T,M,AT,HT,BT,TT} <: AbstractField{B,T,M,AT}
    U::AT # Actual field storing the gauge variables
    halos::HT
    sendbuf::BT
    topology::TT # Info regarding MPI topology
    function Tensorfield{B,T,M}(
        U::AT, halos::HT, sendbuf::BT, topology::TT
    ) where {B,T,M,AT,HT,BT,TT}
        check_types(B, T, U, halos, sendbuf)
        return new{B,T,M,AT,HT,BT,TT}(U, halos, sendbuf, topology)
    end
end

@field_constructor Tensorfield

function Tensorfield(
    u::AbstractField{B,T,M}; no_halo=false, hw=get_halo_width(u)
) where {B,T,M}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Tensorfield{B,T}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Tensorfield{B,T}(size(u)...)
    end

    return u_out
end

const MPITensorfield{B,T} = Tensorfield{B,T,true}

Base.eltype(::Type{Tensorfield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}

# overload get and set for the Tensorfields, so we dont have to do u.U[μ,ν,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Tensorfield, μ, ν, x, y, z, t) =
    u.U[μ, ν, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::Tensorfield, μ, ν, site::SiteCoords) =
    u.U[μ, ν, site]

Base.@propagate_inbounds function Base.getindex(u::MPITensorfield, μ, ν, site::SiteCoords)
    site in u.topology.bulk_sites && return u.U[μ, ν, site]
    ihalo = get_halo_index(site, u.topology.bulk_sites)
    return u.halos[ihalo][μ, ν, site]
end

Base.@propagate_inbounds Base.setindex!(u::Tensorfield, v, μ, ν, x, y, z, t) =
    setindex!(u.U, v, μ, ν, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::Tensorfield, v, μ, ν, site::SiteCoords) =
    setindex!(u.U, v, μ, ν, site)

Base.@propagate_inbounds function Base.setindex!(u::MPITensorfield, v, μ, ν, site::SiteCoords)
    bulk = u.topology.bulk_sites

    if site in bulk
        u.U[μ, ν, site] = v
    else
        ihalo = get_halo_index(site, bulk)
        u.halos[ihalo][μ, ν, site] = v
    end

    return nothing
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
    ::Plaquette, F::Tensorfield{CPU,T}, U::Gaugefield{CPU,T}
) where {T}
    fac = Complex{T}(im)
    update_halo!(U)

    @batch for site in eachindex(U, F)
        C12 = plaquette(U, 1, 2, site)
        F[1, 2, site] = fac * (C12 - C12')
        C13 = plaquette(U, 1, 3, site)
        F[1, 3, site] = fac * (C13 - C13')
        C14 = plaquette(U, 1, 4, site)
        F[1, 4, site] = fac * (C14 - C14')
        C23 = plaquette(U, 2, 3, site)
        F[2, 3, site] = fac * (C23 - C23')
        C24 = plaquette(U, 2, 4, site)
        F[2, 4, site] = fac * (C24 - C24')
        C34 = plaquette(U, 3, 4, site)
        F[3, 4, site] = fac * (C34 - C34')
    end

    return nothing
end

function fieldstrength_eachsite!(
    ::Clover, F::Tensorfield{CPU,T}, U::Gaugefield{CPU,T}
) where {T}
    fac = Complex{T}(im / 8)
    update_halo!(U)

    @batch for site in eachindex(U, F)
        C12 = clover_square(U, 1, 2, site, 1)
        F[1, 2, site] = fac * (C12 - C12')
        C13 = clover_square(U, 1, 3, site, 1)
        F[1, 3, site] = fac * (C13 - C13')
        C14 = clover_square(U, 1, 4, site, 1)
        F[1, 4, site] = fac * (C14 - C14')
        C23 = clover_square(U, 2, 3, site, 1)
        F[2, 3, site] = fac * (C23 - C23')
        C24 = clover_square(U, 2, 4, site, 1)
        F[2, 4, site] = fac * (C24 - C24')
        C34 = clover_square(U, 3, 4, site, 1)
        F[3, 4, site] = fac * (C34 - C34')
    end

    return nothing
end

function create_sendbuf!(F::Tensorfield, sites, dim, dir)
    ibuf = dir + 2(dim - 1)
    sendbuf = F.sendbuf[ibuf]

    @batch for i in eachindex(IndexLinear(), sites)
        site = sites[i]
        
        for ν in 1:4
            for μ in 1:4
                sendbuf[μ, ν, i] = F[μ, ν, site]
            end
        end
    end

    return sendbuf
end

function Base.copyto!(a::Tensorfield, b::Tensorfield, arange, brange)
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    @batch for i in eachindex(IndexLinear(), arange)
        site_a = arange[i]
        site_b = brange[i]

        for ν in 1:4
            for μ in 1:4
                a[μ, ν, site_a] = b[μ, ν, site_b]
            end
        end
    end

    return nothing
end
