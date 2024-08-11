"""
    Fermionfield{BACKEND,T,ND}(NX, NY, NZ, NT)
    Fermionfield(ψ::Fermionfield)
    Fermionfield(f::Abstractfield, staggered)

Creates a Fermionfield on `BACKEND`, i.e. an array of link-variables (NC×ND complex vectors
with `T` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (ND) is reduced to 1 instead of 4.
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
    function Fermionfield{BACKEND,T,A,ND}(U::A, NX, NY, NZ, NT, NV, NC) where {BACKEND,T,A,ND}
        return new{BACKEND,T,A,ND}(U, NX, NY, NZ, NT, NV, NC, ND)
    end

    function Fermionfield{BACKEND,T,ND}(NX, NY, NZ, NT) where {BACKEND,T,ND}
        U = KA.zeros(BACKEND(), SVector{3ND,Complex{T}}, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        return new{BACKEND,T,typeof(U),ND}(U, NX, NY, NZ, NT, NV, 3, ND)
    end

    function Fermionfield(f::Fermionfield{BACKEND,T,A,ND}) where {BACKEND,T,A,ND}
        return Fermionfield{BACKEND,T,ND}(dims(f)...)
    end

    function Fermionfield(f::Abstractfield{BACKEND,T}; staggered=false) where {BACKEND,T}
        ND = staggered ? 1 : 4
        return Fermionfield{BACKEND,T,ND}(dims(f)...)
    end
end

# Need to overload dims and size again, because we are using 4D arrays for fermions
@inline dims(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} =
    NTuple{4,Int64}((size(f, 1), size(f, 2), size(f, 3), size(f, 4)))
Base.size(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = dims(f)
Base.size(f::Fermionfield) = NTuple{4,Int64}((f.NX, f.NY, f.NZ, f.NT))
float_type(::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = T
num_colors(::Fermionfield{B,T,A,ND}) where {B,T,A,ND} = 3
num_dirac(::Fermionfield{B,T,A,ND}) where {B,T,A,ND} = ND
Base.similar(f::Fermionfield) = Fermionfield(f)
Base.eltype(::Fermionfield{B,T}) where {B,T} = Complex{T}
LinearAlgebra.checksquare(f::Fermionfield) = f.NV * num_dirac(f) * num_colors(f)

Base.@propagate_inbounds Base.getindex(f::Fermionfield, i::Integer) = f.U[i]
Base.@propagate_inbounds Base.getindex(f::Fermionfield, x, y, z, t) = f.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::Fermionfield, site::SiteCoords) = f.U[site]
Base.@propagate_inbounds Base.setindex!(f::Fermionfield, v, i::Integer) =
    setindex!(f.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::Fermionfield, v, x, y, z, t) =
    setindex!(f.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::Fermionfield, v, site::SiteCoords) =
    setindex!(f.U, v, site)

function clear!(ϕ::Fermionfield{CPU,T}) where {T}
    @batch for site in eachindex(ϕ)
        ϕ[site] = zero(ϕ[site])
    end

    return nothing
end

function Base.copy!(ϕ::T, ψ::T) where {T<:Fermionfield{CPU}}
    check_dims(ψ, ϕ)

    @batch for site in eachindex(ϕ)
        ϕ[site] = ψ[site]
    end

    return nothing
end

function ones!(ϕ::Fermionfield{CPU,T}) where {T}
    @batch for site in eachindex(ϕ)
        ϕ[site] = fill(1, ϕ[site])
    end

    return nothing
end

function set_source!(ϕ::Fermionfield{CPU,T}, site::SiteCoords, a, μ) where {T}
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:NC
    clear!(ϕ)
    vec_index = (μ - 1) * NC + a
    tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
    ϕ[site] = SVector{3ND,Complex{T}}(tup)
    return nothing
end

function gaussian_pseudofermions!(ϕ::Fermionfield{CPU,T}) where {T}
    sz = num_dirac(ϕ) * num_colors(ϕ)

    for site in eachindex(ϕ)
        ϕ[site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end
end

function LinearAlgebra.mul!(ϕ::Fermionfield{CPU,T}, α) where {T}
    α = T(α)

    @batch for site in eachindex(ϕ)
        ϕ[site] *= α
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ψ::T, ϕ::T) where {T<:Fermionfield{CPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)

    @batch for site in eachindex(ϕ)
        ϕ[site] += α * ψ[site]
    end

    return nothing
end

function LinearAlgebra.axpby!(α, ψ::T, β, ϕ::T) where {T<:Fermionfield{CPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    β = Complex{FloatT}(β)

    @batch for site in eachindex(ϕ)
        ϕ[site] = α * ψ[site] + β * ϕ[site]
    end

    return nothing
end

LinearAlgebra.norm(ϕ::Fermionfield) = sqrt(real(dot(ϕ, ϕ)))

function LinearAlgebra.dot(ϕ::T, ψ::T) where {T<:Fermionfield{CPU}}
    check_dims(ψ, ϕ)
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ)
        res += cdot(ϕ[site], ψ[site])
    end

    return res
end

"""
    even_odd(f::Fermionfield)

Create a wrapper around a `Fermionfield` to signal that it is meant to be used in the context
of even-odd preconditioning. What this amounts to is that we realign the entries such that
`ϕ -> (ϕₑ, ϕₒ)`, which is achieved by recalculating the index whenever we index into `ϕ`
or iterating only over one half of its indices.
"""
even_odd(f::Fermionfield) = EvenOdd(f)

struct EvenOdd{B,T,A,ND} <: Abstractfield{B,T,A}
    parent::Fermionfield{B,T,A,ND}
    function EvenOdd(f::Fermionfield{B,T,A,ND}) where {B,T,A,ND}
        _, _, _, NT = dims(f)
        @assert iseven(NT) "Need even time extent for even-odd preconditioning"
        return new{B,T,A,ND}(f)
    end
end

Fermionfield(f::EvenOdd{B,T,A,ND}) where {B,T,A,ND} = EvenOdd(f.parent)

@inline dims(f::EvenOdd) = dims(f.parent)
Base.size(f::EvenOdd) = size(f.parent)
Base.similar(f::EvenOdd) = even_odd(Fermionfield(f.parent))
Base.eltype(::EvenOdd{B,T}) where {B,T} = Complex{T}
LinearAlgebra.checksquare(f::EvenOdd) = LinearAlgebra.checksquare(f.parent) ÷ 2
num_colors(::EvenOdd{B,T,A,ND}) where {B,T,A,ND} = 3
num_dirac(::EvenOdd{B,T,A,ND}) where {B,T,A,ND} = ND
volume(f::EvenOdd) = volume(f.parent)

Base.@propagate_inbounds Base.getindex(f::EvenOdd, i::Integer) = f.parent.U[i]
Base.@propagate_inbounds Base.getindex(f::EvenOdd, x, y, z, t) = f.parent.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::EvenOdd, site::SiteCoords) = f.parent.U[site]
Base.@propagate_inbounds Base.setindex!(f::EvenOdd, v, i::Integer) =
    setindex!(f.parent.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::EvenOdd, v, x, y, z, t) =
    setindex!(f.parent.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::EvenOdd, v, site::SiteCoords) =
    setindex!(f.parent.U, v, site)


clear!(ϕ_eo::EvenOdd) = clear!(ϕ_eo.parent)
ones!(ϕ_eo::EvenOdd) = ones!(ϕ_eo.parent)

function set_source!(ϕ_eo::EvenOdd{CPU,T}, site::SiteCoords, a, μ) where {T}
    ϕ = ϕ_eo.parent
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:3
    clear!(ϕ)
    vec_index = (μ - 1) * NC + a
    tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
    _site = eo_site(site, dims(ϕ)..., ϕ.NV)
    ϕ[_site] = SVector{3ND,Complex{T}}(tup)
    return nothing
end

function Base.copy!(ϕ_eo::TF, ψ_eo::TF) where {TF<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even = true

    @batch for e_site in eachindex(even, ϕ)
        ϕ[e_site] = ψ[e_site]
    end

    return nothing
end

function gaussian_pseudofermions!(ϕ_eo::EvenOdd{CPU,T}) where {T}
    ϕ = ϕ_eo.parent
    sz = num_dirac(ϕ) * num_colors(ϕ)
    even = true

    for e_site in eachindex(even, ϕ)
        ϕ[e_site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end
end

function LinearAlgebra.mul!(ϕ_eo::EvenOdd{CPU,T}, α) where {T}
    ϕ = ϕ_eo.parent
    α = Complex{T}(α)
    even = true

    @batch for _site in eachindex(even, ϕ)
        ϕ[_site] *= α
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ψ_eo::T, ϕ_eo::T) where {T<:EvenOdd{CPU}} # even on even is the default
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    even = true

    @batch for _site in eachindex(even, ϕ)
        ϕ[_site] += α * ψ[_site]
    end

    return nothing
end

function LinearAlgebra.axpby!(α, ψ_eo::T, β, ϕ_eo::T, even=true) where {T<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    β = Complex{FloatT}(β)

    @batch for _site in eachindex(even, ϕ)
        ϕ[_site] = α * ψ[_site] + β * ϕ[_site]
    end

    return nothing
end

LinearAlgebra.norm(ϕ_eo::EvenOdd) = sqrt(real(dot(ϕ_eo, ϕ_eo)))

function LinearAlgebra.dot(ϕ_eo::T, ψ_eo::T) where {T<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision
    even = true

    @batch reduction = (+, res) for _site in eachindex(even, ϕ)
        res += cdot(ϕ[_site], ψ[_site])
    end

    return res
end

function dot_all(ϕ_eo::T, ψ_eo::T) where {T<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ)
        res += cdot(ϕ[site], ψ[site])
    end

    return res
end

function copy_eo!(ϕ_eo::T, ψ_eo::T) where {T<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV
    even = true

    for e_site in eachindex(even, ϕ)
        o_site = switch_sides(e_site, fdims..., NV)
        ϕ[e_site] = ψ[o_site]
    end

    return nothing
end

function copy_oe!(ϕ_eo::T, ψ_eo::T) where {T<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV
    odd = false

    for o_site in eachindex(odd, ϕ)
        e_site = switch_sides(o_site, fdims..., NV)
        ϕ[o_site] = ψ[e_site]
    end

    return nothing
end

function axpy_oe!(α, ψ_eo::T, ϕ_eo::T) where {T<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    fdims = dims(ϕ)
    NV = ϕ.NV
    even = true

    for e_site in eachindex(even, ϕ)
        o_site = switch_sides(e_site, fdims..., NV)
        ϕ[e_site] += α * ψ[o_site]
    end

    return nothing
end

function axpy_eo!(α, ψ_eo::T, ϕ_eo::T) where {T<:EvenOdd{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    fdims = dims(ϕ)
    NV = ϕ.NV
    odd = false

    for o_site in eachindex(odd, ϕ)
        e_site = switch_sides(o_site, fdims..., NV)
        ϕ[o_site] += α * ψ[e_site]
    end

    return nothing
end
