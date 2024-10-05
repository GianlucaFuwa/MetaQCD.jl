"""
    Spinorfield{Backend,FloatType,NumDirac}(NX, NY, NZ, NT)
    Spinorfield(ψ::Spinorfield)
    Spinorfield(f::AbstractField, staggered)

Creates a Spinorfield on `Backend`, i.e. an array of link-variables (numcolors×NumDirac complex vectors
with `FloatType` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (NumDirac) is reduced to 1 instead of 4.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Spinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    
    topology::FieldTopology # Info regarding MPI topology
    function Spinorfield{Backend,FloatType,NumDirac}(
        NX, NY, NZ, NT
    ) where {Backend,FloatType,NumDirac}
        U = KA.zeros(Backend(), SVector{3NumDirac,Complex{FloatType}}, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        numprocs_cart = (1, 1, 1, 1)
        halo_width = 0
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        return new{Backend,FloatType,false,typeof(U),NumDirac}(
            U, NX, NY, NZ, NT, NV, 3, topology
        )
    end

    function Spinorfield{Backend,FloatType,NumDirac}(
        NX, NY, NZ, NT, numprocs_cart, halo_width
    ) where {Backend,FloatType,NumDirac}
        if prod(numprocs_cart) == 1
            return Spinorfield{Backend,FloatType,NumDirac}(NX, NY, NZ, NT)
        end

        NV = NX * NY * NZ * NT
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        ldims = topology.local_dims
        dims_in = ntuple(i -> ldims[i]+2halo_width, Val(4)) 
        U = KA.zeros(Backend(), SVector{3NumDirac,Complex{FloatType}}, dims_in...)
        return new{Backend,FloatType,true,typeof(U),NumDirac}(
            U, NX, NY, NZ, NT, NV, 3, topology
        )
    end
end

function Spinorfield(
    f::Spinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac}
) where {Backend,FloatType,IsDistributed,ArrayType,NumDirac}
    u_out = if IsDistributed
        numprocs_cart = f.topology.numprocs_cart
        halo_width = f.topology.halo_width
        Spinorfield{Backend,FloatType,NumDirac}(f.NX, f.NY, f.NZ, f.NT, numprocs_cart, halo_width)
    else
        Spinorfield{Backend,FloatType,NumDirac}(f.NX, f.NY, f.NZ, f.NT)
    end

    return u_out
end

function Spinorfield(
    u::AbstractField{Backend,FloatType,IsDistributed}; staggered=false
) where {Backend,FloatType,IsDistributed}
    NumDirac = staggered ? 1 : 4

    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = u.topology.halo_width
        Spinorfield{Backend,FloatType,NumDirac}(u.NX, u.NY, u.NZ, u.NT, numprocs_cart, halo_width)
    else
        Spinorfield{Backend,FloatType,NumDirac}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
end

# Need to overload dims and size again, because we are using 4D arrays for fermions
@inline dims(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} =
    NTuple{4,Int64}((size(f, 1), size(f, 2), size(f, 3), size(f, 4)))
@inline Base.size(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = dims(f)
@inline Base.size(f::Spinorfield) = NTuple{4,Int64}((f.NX, f.NY, f.NZ, f.NT))
@inline float_type(::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = T
num_colors(::Spinorfield{B,T,M,A,ND}) where {B,T,M,A,ND} = 3
num_dirac(::Spinorfield{B,T,M,A,ND}) where {B,T,M,A,ND} = ND
Base.similar(f::Spinorfield) = Spinorfield(f)
Base.eltype(::Spinorfield{B,T}) where {B,T} = Complex{T}
LinearAlgebra.checksquare(f::Spinorfield) = f.NV * num_dirac(f) * num_colors(f)

Base.@propagate_inbounds Base.getindex(f::Spinorfield, i::Integer) = f.U[i]
Base.@propagate_inbounds Base.getindex(f::Spinorfield, x, y, z, t) = f.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::Spinorfield, site::SiteCoords) = f.U[site]
Base.@propagate_inbounds Base.setindex!(f::Spinorfield, v, i::Integer) =
    setindex!(f.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::Spinorfield, v, x, y, z, t) =
    setindex!(f.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::Spinorfield, v, site::SiteCoords) =
    setindex!(f.U, v, site)

Base.view(f::Spinorfield, I::CartesianIndices{4}) = view(f.U, I.indices...)

function clear!(ϕ::Spinorfield{CPU,T}) where {T}
    @batch for site in allindices(ϕ)
        ϕ[site] = zero(ϕ[site])
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function Base.copy!(ϕ::T, ψ::T) where {T<:Spinorfield{CPU}}
    check_dims(ψ, ϕ)

    @batch for site in allindices(ϕ)
        ϕ[site] = ψ[site]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    # We assume that ψ's halo is already up-to-date before calling this
    return nothing
end

function ones!(ϕ::Spinorfield{CPU,T}) where {T}
    @batch for site in allindices(ϕ)
        ϕ[site] = fill(1, ϕ[site])
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function set_source!(ϕ::Spinorfield{CPU,T}, site::SiteCoords, a, μ) where {T}
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:NC
    clear!(ϕ)
    vec_index = (μ - 1) * NC + a
    tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
    ϕ[site] = SVector{3ND,Complex{T}}(tup)
    update_halo!(ϕ)
    return nothing
end

function gaussian_pseudofermions!(ϕ::Spinorfield{CPU,T}) where {T}
    sz = num_dirac(ϕ) * num_colors(ϕ)

    for site in eachindex(ϕ)
        ϕ[site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end

    update_halo!(ϕ)
    return nothing
end

function LinearAlgebra.mul!(ϕ::Spinorfield{CPU,T}, α) where {T}
    α = T(α)

    @batch for site in allindices(ϕ)
        ϕ[site] *= α
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    # We assume that ϕ's halo is already up-to-date before calling this
    return nothing
end

function LinearAlgebra.axpy!(α, ψ::T, ϕ::T) where {T<:Spinorfield{CPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)

    # I'm pretty sure iterating over all indices is fine here
    @batch for site in allindices(ϕ)
        ϕ[site] += α * ψ[site]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function LinearAlgebra.axpby!(α, ψ::T, β, ϕ::T) where {T<:Spinorfield{CPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    β = Complex{FloatT}(β)

    # I'm pretty sure iterating over all indices is fine here
    @batch for site in allindices(ϕ)
        ϕ[site] = α * ψ[site] + β * ϕ[site]
    end

    return nothing
end

LinearAlgebra.norm(ϕ::Spinorfield) = sqrt(real(dot(ϕ, ϕ)))

function LinearAlgebra.dot(ϕ::T, ψ::T) where {T<:Spinorfield{CPU}}
    check_dims(ψ, ϕ)
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ)
        res += cdot(ϕ[site], ψ[site])
    end

    return distributed_reduce(res, +, ϕ)
end
