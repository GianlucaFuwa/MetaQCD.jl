"""
    Spinorfield{Backend,T,ND}(NX, NY, NZ, NT)
    Spinorfield(ψ::Spinorfield)
    Spinorfield(f::AbstractField, staggered)

Creates a Spinorfield on `Backend`, i.e. an array of link-variables (NC×ND complex vectors
with `T` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (ND) is reduced to 1 instead of 4.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Spinorfield{B,T,M,A,H,ND} <: AbstractField{B,T,M,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    halo_sendbuf::H
    halo_recvbuf::H
    pad::Int64 # halo width
    
    # Meta-data regarding portion of the field local to current process
    my_NX::Int64
    my_NY::Int64
    my_NZ::Int64
    my_NT::Int64
    my_NV::Int64
    nprocs::Int64
    nprocs_cart::NTuple{4,Int64}
    myrank::Int64
    myrank_cart::NTuple{4,Int64}
    comm_cart::MPI.Comm
    # TODO: Constructors
end

# Need to overload dims and size again, because we are using 4D arrays for fermions
local_dims(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} =
    NTuple{4,Int64}((size(f, 1), size(f, 2), size(f, 3), size(f, 4)))
Base.size(f::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = dims(f)
Base.size(f::Spinorfield) = NTuple{4,Int64}((f.NX, f.NY, f.NZ, f.NT))
float_type(::AbstractArray{SVector{N,Complex{T}},4}) where {N,T} = T
num_colors(::Spinorfield{B,T,A,ND}) where {B,T,A,ND} = 3
num_dirac(::Spinorfield{B,T,A,ND}) where {B,T,A,ND} = ND
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

function clear!(ϕ::Spinorfield{CPU,T}) where {T}
    @batch for site in eachindex(ϕ)
        ϕ[site] = zero(ϕ[site])
    end

    return nothing
end

function Base.copy!(ϕ::T, ψ::T) where {T<:Spinorfield{CPU}}
    check_dims(ψ, ϕ)

    @batch for site in eachindex(ϕ)
        ϕ[site] = ψ[site]
    end

    return nothing
end

function ones!(ϕ::Spinorfield{CPU,T}) where {T}
    @batch for site in eachindex(ϕ)
        ϕ[site] = fill(1, ϕ[site])
    end

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
    return nothing
end

function gaussian_pseudofermions!(ϕ::Spinorfield{CPU,T}) where {T}
    sz = num_dirac(ϕ) * num_colors(ϕ)

    for site in eachindex(ϕ)
        ϕ[site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end
end

function LinearAlgebra.mul!(ϕ::Spinorfield{CPU,T}, α) where {T}
    α = T(α)

    @batch for site in eachindex(ϕ)
        ϕ[site] *= α
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ψ::T, ϕ::T) where {T<:Spinorfield{CPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)

    @batch for site in eachindex(ϕ)
        ϕ[site] += α * ψ[site]
    end

    return nothing
end

function LinearAlgebra.axpby!(α, ψ::T, β, ϕ::T) where {T<:Spinorfield{CPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    β = Complex{FloatT}(β)

    @batch for site in eachindex(ϕ)
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

    return res
end

"""
    even_odd(f::Spinorfield)

Create a wrapper around a `Spinorfield` to signal that it is meant to be used in the context
of even-odd preconditioning. What this amounts to is that we realign the entries such that
`ϕ -> (ϕₑ, ϕₒ)`, which is achieved by recalculating the index whenever we index into `ϕ`
or iterating only over one half of its indices.
"""
even_odd(f::Spinorfield) = SpinorfieldEO(f)

struct SpinorfieldEO{B,T,M,A,H,ND} <: AbstractField{B,T,M,A}
    parent::Spinorfield{B,T,M,A,H,ND}
    function SpinorfieldEO(f::Spinorfield{B,T,M,A,H,ND}) where {B,T,M,A,H,ND}
        @assert iseven(f.NT) "Need even time extent for even-odd preconditioning"
        return new{B,T,M,A,H,ND}(f)
    end
end

Spinorfield(f::SpinorfieldEO{B,T,M,A,ND}) where {B,T,M,A,ND} = SpinorfieldEO(f.parent)

local_dims(f::SpinorfieldEO) = local_dims(f.parent)
Base.size(f::SpinorfieldEO) = size(f.parent)
Base.similar(f::SpinorfieldEO) = even_odd(Spinorfield(f.parent))
Base.eltype(::SpinorfieldEO{B,T}) where {B,T} = Complex{T}
LinearAlgebra.checksquare(f::SpinorfieldEO) = LinearAlgebra.checksquare(f.parent) ÷ 2
num_colors(::SpinorfieldEO{B,T,M,A,ND}) where {B,T,M,A,ND} = 3
num_dirac(::SpinorfieldEO{B,T,M,A,ND}) where {B,T,M,A,ND} = ND
volume(f::SpinorfieldEO) = volume(f.parent)

Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, i::Integer) = f.parent.U[i]
Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, x, y, z, t) = f.parent.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, site::SiteCoords) = f.parent.U[site]
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, i::Integer) =
    setindex!(f.parent.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, x, y, z, t) =
    setindex!(f.parent.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, site::SiteCoords) =
    setindex!(f.parent.U, v, site)


clear!(ϕ_eo::SpinorfieldEO) = clear!(ϕ_eo.parent)
ones!(ϕ_eo::SpinorfieldEO) = ones!(ϕ_eo.parent)

function set_source!(ϕ_eo::SpinorfieldEO{CPU,T}, site::SiteCoords, a, μ) where {T}
    ϕ = ϕ_eo.parent
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:3
    clear!(ϕ)
    vec_index = (μ - 1) * NC + a
    tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
    _site = eo_site(site, global_dims(ϕ)..., ϕ.NV)
    ϕ[_site] = SVector{3ND,Complex{T}}(tup)
    return nothing
end

function Base.copy!(ϕ_eo::TF, ψ_eo::TF) where {TF<:SpinorfieldEO{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even = true

    @batch for e_site in eachindex(even, ϕ)
        ϕ[e_site] = ψ[e_site]
    end

    return nothing
end

function gaussian_pseudofermions!(ϕ_eo::SpinorfieldEO{CPU,T}) where {T}
    ϕ = ϕ_eo.parent
    sz = num_dirac(ϕ) * num_colors(ϕ)
    even = true

    for e_site in eachindex(even, ϕ)
        ϕ[e_site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end
end

function LinearAlgebra.mul!(ϕ_eo::SpinorfieldEO{CPU,T}, α) where {T}
    ϕ = ϕ_eo.parent
    α = Complex{T}(α)
    even = true

    @batch for _site in eachindex(even, ϕ)
        ϕ[_site] *= α
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ψ_eo::T, ϕ_eo::T) where {T<:SpinorfieldEO{CPU}} # even on even is the default
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

function LinearAlgebra.axpby!(α, ψ_eo::T, β, ϕ_eo::T, even=true) where {T<:SpinorfieldEO{CPU}}
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

LinearAlgebra.norm(ϕ_eo::SpinorfieldEO) = sqrt(real(dot(ϕ_eo, ϕ_eo)))

function LinearAlgebra.dot(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
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

function dot_all(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ)
        res += cdot(ϕ[site], ψ[site])
    end

    return res
end

function copy_eo!(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
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

function copy_oe!(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
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

function axpy_oe!(α, ψ_eo::T, ϕ_eo::T) where {T<:SpinorfieldEO{CPU}}
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

function axpy_eo!(α, ψ_eo::T, ϕ_eo::T) where {T<:SpinorfieldEO{CPU}}
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
