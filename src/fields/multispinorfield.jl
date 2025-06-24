"""
    MultiSpinorfield{B,T,ND}(NX, NY, NZ, NT, numspinors)
    MultiSpinorfield(ψ::MultiSpinorfield)
    MultiSpinorfield(f::AbstractField; numspinors=1, staggered=false)

Creates a MultiSpinorfield on `B`, i.e. an array of link-variables (numcolors×NumDirac×numspinors complex vectors
with `T` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (ND) is reduced to 1 instead of 4.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct MultiSpinorfield{B,T,M,AT,GA,HT,BT,TT} <: AbstractField{B,T,M,AT}
    U::AT # Actual field storing the gauge variables
    halos::HT
    sendbuf::BT
    topology::TT # Info regarding MPI topology
    numspinors::Int64
    function MultiSpinorfield{B,T,M,ND}(
        U::AT, halos::HT, sendbuf::BT, topology::TT, numspinors
    ) where {B,T,M,AT,ND,HT,BT,TT}
        check_types(B, T, U, halos, sendbuf)
        return new{B,T,M,AT,ND,HT,BT,TT}(U, halos, sendbuf, topology, numspinors)
    end
end

@field_constructor MultiSpinorfield extra_types=ND extra_args=numspinors

function MultiSpinorfield(
    f::MultiSpinorfield{B,T,M,ND}; no_halo=false, hw=halo_width(f)
) where {B,T,M,ND}
    global_dims = f.topology.global_dims

    u_out = if M
        ncart = f.topology.numprocs_cart
        MultiSpinorfield{B,T,ND}(
            global_dims..., f.numspinors;
            numprocs_cart=ncart, halo_width=hw, no_halo=no_halo
        )
    else
        MultiSpinorfield{B,T,ND}(global_dims..., f.numspinors)
    end

    return u_out
end

function MultiSpinorfield(
    u::AbstractField{B,T,M}, numspinors; staggered=false, no_halo=false, hw=get_halo_width(u)
) where {B,T,M}
    ND = if u isa Spinorfield
        num_dirac(u)
    else
        staggered ? 1 : 4
    end

    u_out = if M
        ncart = get_numprocs_cart(u)
        MultiSpinorfield{B,T,ND}(
            size(u)..., numspinors;
            numprocs_cart=ncart, halo_width=hw, no_halo=no_halo
        )
    else
        MultiSpinorfield{B,T,ND}(size(u)..., numspinors)
    end

    return u_out
end

const MPIMultiSpinorfield{B,T,AT,ND,HT,TT} = MultiSpinorfield{B,T,true,AT,ND,HT,TT}

# Need to overload dims and size again, because we are using 4D arrays for fermions
@inline num_dirac(::MultiSpinorfield{B,T,M,A,ND}) where {B,T,M,A,ND} = ND
@inline num_spinors(f::MultiSpinorfield) = f.numspinors
LinearAlgebra.checksquare(f::MultiSpinorfield) = length(f) * num_dirac(f) * num_colors(f)
function Base.eltype(::Type{MultiSpinorfield}, ::Type{T}, ::Val{ND}) where {T,ND}
    return SVector{3ND,Complex{T}}
end

Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, i::Integer) = f.U[i]
Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, s, x, y, z, t) = f.U[s, x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, s, site::SiteCoords) = f.U[s, site]

Base.@propagate_inbounds function Base.getindex(u::MPIMultiSpinorfield, is, site::SiteCoords)
    return _getindex_lat(u, is, site, u.topology.bulk_sites, u.topology.halo_width)
end

Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, i::Integer) =
    setindex!(f.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, s, x, y, z, t) =
    setindex!(f.U, v, s, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, s, site::SiteCoords) =
    setindex!(f.U, v, s, site)

Base.@propagate_inbounds function Base.setindex!(u::MPIMultiSpinorfield, v, is, site::SiteCoords)
    return _setindex_lat!(u, v, is, site, u.topology.bulk_sites, u.topology.halo_width)
end

function clear!(ϕ::MultiSpinorfield{CPU,T}) where {T}
    @batch for site in eachindex(ϕ)
        for is in 1:ϕ.numspinors
            ϕ[is, site] = zero(ϕ[is, site])
        end
    end

    return nothing
end

function Base.copy!(ϕ::T, ψ::T) where {T<:MultiSpinorfield{CPU}}
    check_dims(ψ, ϕ)
    @assert ψ.numspinors == ϕ.numspinors

    @batch for site in eachindex(ϕ)
        for is in 1:ϕ.numspinors
            ϕ[is, site] = ψ[is, site]
        end
    end

    return nothing
end

function ones!(ϕ::MultiSpinorfield{CPU,T}) where {T}
    @batch for site in eachindex(ϕ)
        for is in 1:ϕ.numspinors
            ϕ[is, site] = fill(1, ϕ[is, site])
        end
    end

    return nothing
end

function create_sendbuf!(f::MultiSpinorfield, sites, dim, dir)
    numspinors = num_spinors(f)
    ibuf = dir + 2(dim - 1)
    sendbuf = f.sendbuf[ibuf]

    @batch for i in eachindex(IndexLinear(), sites)
        site = sites[i]
        
        for is in 1:numspinors
            sendbuf[is, i] = f[is, site]
        end
    end

    return sendbuf
end

function Base.copyto!(a::MultiSpinorfield, b::MultiSpinorfield, arange, brange)
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"
    @assert num_spinors(a) == num_spinors(b) "input fields must have same `numspinors`"
    numspinors = num_spinors(a)

    @batch for i in eachindex(IndexLinear(), arange)
        site_a = arange[i]
        site_b = brange[i]

        for is in 1:numspinors
            a[is, site_a] = b[is, site_b]
        end
    end

    return nothing
end
