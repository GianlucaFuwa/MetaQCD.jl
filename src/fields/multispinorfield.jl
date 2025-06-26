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

function ones!(ϕ::MultiSpinorfield{B,T}) where {B,T}
    numspinors = num_spinors(ϕ)

    parallelfor(eachindex(ϕ), B) do site
        for is in 1:numspinors
            ϕ[is, site] = fill(1, ϕ[is, site])
        end
    end

    return nothing
end

function set_source!(ϕ::MultiSpinorfield{B,T}, source::SiteCoords, a, μ) where {B,T}
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:3
    vec_index = 3(μ - 1) + a
    numspinors = ϕ.numspinors

    parallelfor(eachindex(ϕ), B) do site
        if site == source
            tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
            for is in 1:numspinors
                ϕ[is, site] = SVector{3ND,Complex{T}}(tup)
            end
        else
            for is in 1:numspinors
                ϕ[is, site] = zero(SVector{3ND,Complex{T}})
            end
        end
    end

    return nothing
end

function gaussian_pseudofermions!(ϕ::MultiSpinorfield{B,T}) where {B,T}
    ND = num_dirac(ϕ)
    numspinors = ϕ.numspinors

    parallelfor(eachindex(ϕ), B) do site
        for is in 1:numspinors
            ϕ[is, site] = randn(SVector{3ND,Complex{T}}) # σ = 0.5
        end
    end

    return nothing
end

function create_sendbuf!(f::MultiSpinorfield{B}, sites, dim, dir) where {B}
    numspinors = num_spinors(f)
    ibuf = dir + 2(dim - 1)
    sendbuf = f.sendbuf[ibuf]

    parallelfor(eachindex(IndexLinear(), sites), B) do i
        site = sites[i]

        for is in 1:numspinors
            sendbuf[is, i] = f[is, site]
        end
    end

    return sendbuf
end

function Base.copyto!(a::T, b::T, arange, brange) where {B,T<:MultiSpinorfield{B}}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"
    @assert num_spinors(a) == num_spinors(b) "input fields must have same `numspinors`"
    numspinors = num_spinors(a)

    parallelfor(eachindex(IndexLinear(), arange), B) do i
        site_a = arange[i]
        site_b = brange[i]

        for is in 1:numspinors
            a[is, site_a] = b[is, site_b]
        end
    end

    return nothing
end
