"""
    MultiSpinorfield{Backend,FloatType,NumDirac}(NX, NY, NZ, NT, NS)
    MultiSpinorfield(ψ::MultiSpinorfield)
    MultiSpinorfield(f::AbstractField; NS=1, staggered=false)

Creates a MultiSpinorfield on `Backend`, i.e. an array of link-variables (numcolors×NumDirac×numspinors complex vectors
with `FloatType` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (NumDirac) is reduced to 1 instead of 4.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct MultiSpinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NS::Int64 # Number of Spinors
    NC::Int64 # Number of colors
    
    topology::FieldTopology # Info regarding MPI topology
    function MultiSpinorfield{Backend,FloatType,NumDirac}(
        NX, NY, NZ, NT, NS
    ) where {Backend,FloatType,NumDirac}
        U = KA.zeros(Backend(), SVector{3NumDirac,Complex{FloatType}}, NS, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        numprocs_cart = (1, 1, 1, 1)
        halo_width = 0
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        return new{Backend,FloatType,false,typeof(U),NumDirac}(
            U, NX, NY, NZ, NT, NV, NS, 3, topology
        )
    end

    function MultiSpinorfield{Backend,FloatType,NumDirac}(
        NX, NY, NZ, NT, NS, numprocs_cart, halo_width
    ) where {Backend,FloatType,NumDirac}
        if prod(numprocs_cart) == 1
            return MultiSpinorfield{Backend,FloatType,NumDirac}(NX, NY, NZ, NT, NS)
        end

        NV = NX * NY * NZ * NT
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        ldims = topology.local_dims
        dims_in = ntuple(i -> ldims[i]+2halo_width, Val(4)) 
        U = KA.zeros(Backend(), SVector{3NumDirac,Complex{FloatType}}, NS, dims_in...)
        return new{Backend,FloatType,true,typeof(U),NumDirac}(
            U, NX, NY, NZ, NT, NV, NS, 3, topology
        )
    end
end

function MultiSpinorfield(
    f::MultiSpinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac}
) where {Backend,FloatType,IsDistributed,ArrayType,NumDirac}
    u_out = if IsDistributed
        numprocs_cart = f.topology.numprocs_cart
        halo_width = f.topology.halo_width
        MultiSpinorfield{Backend,FloatType,NumDirac}(
            f.NX, f.NY, f.NZ, f.NT, f.NS, numprocs_cart, halo_width
        )
    else
        MultiSpinorfield{Backend,FloatType,NumDirac}(f.NX, f.NY, f.NZ, f.NT, f.NS)
    end

    return u_out
end

function MultiSpinorfield(
    u::AbstractField{Backend,FloatType,IsDistributed}; NS=1, staggered=false
) where {Backend,FloatType,IsDistributed}
    NumDirac = staggered ? 1 : 4

    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = u.topology.halo_width
        MultiSpinorfield{Backend,FloatType,NumDirac}(
            u.NX, u.NY, u.NZ, u.NT, NS, numprocs_cart, halo_width
        )
    else
        MultiSpinorfield{Backend,FloatType,NumDirac}(u.NX, u.NY, u.NZ, u.NT, NS)
    end

    return u_out
end

# Need to overload dims and size again, because we are using 4D arrays for fermions
@inline dims(f::AbstractArray{SVector{N,Complex{T}},5}) where {N,T} =
    NTuple{4,Int64}((size(f, 2), size(f, 3), size(f, 4), size(f, 5)))
@inline Base.size(f::AbstractArray{SVector{N,Complex{T}},5}) where {N,T} = dims(f)
@inline Base.size(f::MultiSpinorfield) = NTuple{4,Int64}((f.NX, f.NY, f.NZ, f.NT))
@inline float_type(::AbstractArray{SVector{N,Complex{T}},5}) where {N,T} = T
num_colors(::MultiSpinorfield{B,T,M,A,ND}) where {B,T,M,A,ND} = 3
num_dirac(::MultiSpinorfield{B,T,M,A,ND}) where {B,T,M,A,ND} = ND
num_spinors(f::MultiSpinorfield) = f.NS
Base.similar(f::MultiSpinorfield) = MultiSpinorfield(f)
Base.eltype(::MultiSpinorfield{B,T}) where {B,T} = Complex{T}
LinearAlgebra.checksquare(f::MultiSpinorfield) = f.NV * num_dirac(f) * num_colors(f)

Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, i::Integer) = f.U[i]
Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, s, x, y, z, t) = f.U[s, x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, s, site::SiteCoords) = f.U[s, site]
Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, i::Integer) =
    setindex!(f.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, s, x, y, z, t) =
    setindex!(f.U, v, s, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, s, site::SiteCoords) =
    setindex!(f.U, v, s, site)

Base.view(f::MultiSpinorfield, s, I::CartesianIndices{4}) = view(f.U, s, I.indices...)

function clear!(ϕ::MultiSpinorfield{CPU,T}) where {T}
    @batch for site in allindices(ϕ)
        for is in 1:ϕ.NS
            ϕ[is, site] = zero(ϕ[is, site])
        end
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function Base.copy!(ϕ::T, ψ::T) where {T<:MultiSpinorfield{CPU}}
    check_dims(ψ, ϕ)
    @assert ψ.NS == ϕ.NS

    @batch for site in allindices(ϕ)
        for is in 1:ϕ.NS
            ϕ[is, site] = ψ[is, site]
        end
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    # We assume that ψ's halo is already up-to-date before calling this
    return nothing
end

function ones!(ϕ::MultiSpinorfield{CPU,T}) where {T}
    @batch for site in allindices(ϕ)
        for is in 1:ϕ.NS
            ϕ[is, site] = fill(1, ϕ[is, site])
        end
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end
