"""
	Fermionfield(NX, NY, NZ, NT; BACKEND=CPU, T=Float64, staggered=false)
	Fermionfield(ψ::Fermionfield)
    Fermionfield(u::Abstractfield, staggered)

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
        U = KA.zeros(BACKEND(), SVector{3,Complex{T}}, ND, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        return new{BACKEND,T,typeof(U),ND}(U, NX, NY, NZ, NT, NV, 3, ND)
    end
end

function Fermionfield(f::Fermionfield{BACKEND,T,A,ND}) where {BACKEND,T,A,ND}
    staggered = ND == 1
    return Fermionfield(dims(f)...; BACKEND=BACKEND, T=T, staggered=staggered)
end

function Fermionfield(f::Abstractfield{BACKEND,T}, staggered) where {BACKEND,T}
    return Fermionfield(dims(f)...; BACKEND=BACKEND, T=T, staggered=staggered)
end

float_type(::AbstractArray{SVector{3,Complex{T}},5}) where {T} = T
num_dirac(::Fermionfield{B,T,A,ND}) where {B,T,A,ND} = ND
Base.similar(f::Fermionfield) = Fermionfield(f)

function clear!(ψ::Fermionfield{CPU,T}) where {T}
    ND = ψ.ND

    @batch for site in eachindex(ψ)
        for μ in 1:ND
            ψ[μ, site] = zerov3(T)
        end
    end

    return nothing
end

function ones!(ψ::Fermionfield{CPU,T}) where {T}
    ND = ψ.ND

    @batch for site in eachindex(ψ)
        for μ in 1:ND
            ψ[μ, site] = onev3(T)
        end
    end

    return nothing
end

function Base.copy!(ϕ::T, ψ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)
    ND = ϕ.ND

    @batch for site in eachindex(ϕ)
        for μ in 1:ND
            ϕ[μ, site] = ψ[μ, site]
        end
    end

    return nothing
end

function gaussian_pseudofermions!(f::Fermionfield{CPU,T}) where {T}
    ND = f.ND

    @batch for site in eachindex(f)
        for μ in 1:ND
            f[μ, site] = @SVector randn(Complex{T}, 3) # σ = 0.5
        end
    end
end

function LinearAlgebra.axpy!(α, ψ::T, ϕ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)
    ND = ϕ.ND
    α = float_type(ϕ)(α)

    @batch for site in eachindex(ϕ)
        for μ in 1:ND
            ϕ[μ, site] += α * ψ[μ, site]
        end
    end

    return nothing
end

function LinearAlgebra.axpby!(α, ψ::T, β, ϕ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)
    ND = ϕ.ND
    α = float_type(ϕ)(α)
    β = float_type(ϕ)(β)

    @batch for site in eachindex(ϕ)
        for μ in 1:ND
            ϕ[μ, site] = α * ψ[μ, site] + β * ϕ[μ, site]
        end
    end

    return nothing
end

function LinearAlgebra.dot(ϕ::T, ψ::T) where {T<:Fermionfield{CPU}}
    @assert dims(ϕ) == dims(ψ)
    ND = ϕ.ND
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ)
        for μ in 1:ND
            res += cdot(ϕ[μ, site], ψ[μ, site])
        end
    end

    return res
end