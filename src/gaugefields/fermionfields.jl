"""
	Fermionfield(NX, NY, NZ, NT; BACKEND=CPU, T=Float64, staggered=false)
	Fermionfield(ψ::Fermionfield)
    Fermionfield(f::Fermionfield, staggered)

Creates a Fermionfield on `BACKEND`, i.e. an array of link-variables (3D complex vectors
with `T` precision) of size `ND × NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom is reduced to 1 instead of 4.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Fermionfield{BACKEND,T,A,ND} <: Abstractfield{BACKEND,T,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    ND::Int64 # Number of dirac indices
    function Fermionfield(NX, NY, NZ, NT; BACKEND=CPU, T=Float64, staggered=false)
        @assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
        ND = staggered ? 1 : 4
        U = KA.zeros(BACKEND(), SVector{3ND,Complex{T}}, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        return new{BACKEND,T,typeof(U),ND}(U, NX, NY, NZ, NT, NV, 3, ND)
    end
end

function Fermionfield(f::Fermionfield{BACKEND,T,A,ND}) where {BACKEND,T,A,ND}
    staggered = ND == 1
    return Fermionfield(dims(f)...; BACKEND=BACKEND, T=T, staggered=staggered)
end

function Fermionfield(f::Abstractfield{BACKEND,T}; staggered=false) where {BACKEND,T}
    return Fermionfield(dims(f)...; BACKEND=BACKEND, T=T, staggered=staggered)
end

# Need to overload dims and size again, because we are using 4D arrays for fermions
@inline dims(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} =
    NTuple{4,Int64}((size(f, 1), size(f, 2), size(f, 3), size(f, 4)))
Base.size(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = dims(f)
Base.size(f::Fermionfield) = NTuple{4,Int64}((f.NX, f.NY, f.NZ, f.NT))
float_type(::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = T
num_dirac(::Fermionfield{B,T,A,ND}) where {B,T,A,ND} = ND
Base.similar(f::Fermionfield) = Fermionfield(f)

Base.@propagate_inbounds Base.getindex(f::Fermionfield, x, y, z, t) = f.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::Fermionfield, site::SiteCoords) = f.U[site]
Base.@propagate_inbounds Base.setindex!(f::Fermionfield, v, x, y, z, t) =
    setindex!(f.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::Fermionfield, v, site::SiteCoords) =
    setindex!(f.U, v, site)

function clear!(ψ::Fermionfield{CPU,T}) where {T}
    @batch for site in eachindex(ψ)
        ψ[site] = zero(ψ[site])
    end

    return nothing
end

function ones!(ψ::Fermionfield{CPU,T}) where {T}
    @batch for site in eachindex(ψ)
        ψ[site] = fill(1, ψ[site])
    end

    return nothing
end

function set_source!(ψ::Fermionfield{CPU,T}, site::SiteCoords, a, μ) where {T}
    ND = num_dirac(ψ)
    @assert μ ∈ 1:ND && a ∈ 1:3
    clear!(ψ)
    vec_index = (μ - 1) * ND + a
    tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), 3ND)
    ψ[site] = SVector{3ND,Complex{T}}(tup)
    return nothing
end

function Base.copy!(ϕ::T, ψ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)

    @batch for site in eachindex(ϕ)
        ϕ[site] = ψ[site]
    end

    return nothing
end

function gaussian_pseudofermions!(f::Fermionfield{CPU,T}) where {T}
    sz = f.ND * f.NC

    for site in eachindex(f)
        f[site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end
end

function LinearAlgebra.axpy!(α, ψ::T, ϕ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)
    α = float_type(ϕ)(α)

    @batch for site in eachindex(ϕ)
        ϕ[site] += α * ψ[site]
    end

    return nothing
end

function LinearAlgebra.axpby!(α, ψ::T, β, ϕ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)
    α = float_type(ϕ)(α)
    β = float_type(ϕ)(β)

    @batch for site in eachindex(ϕ)
        ϕ[site] = α * ψ[site] + β * ϕ[site]
    end

    return nothing
end

function LinearAlgebra.dot(ϕ::T, ψ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ)
        res += cdot(ϕ[site], ψ[site])
    end

    return res
end
