"""
    even_odd(f::Spinorfield)

Create a wrapper around a `Spinorfield` to signal that it is meant to be used in the context
of even-odd preconditioning. What this amounts to is that we realign the entries such that
`ϕ -> (ϕₑ, ϕₒ)`, which is achieved by recalculating the index whenever we index into `ϕ`
or iterating only over one half of its indices.
"""
even_odd(f::Spinorfield) = SpinorfieldEO(f)

struct SpinorfieldEO{Backend,FloatType,IsDistributed,ArrayType,NumDirac} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    parent::Spinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac}
    function SpinorfieldEO(
        f::Spinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac}
    ) where {Backend,FloatType,IsDistributed,ArrayType,NumDirac}
        @assert iseven(f.NT) "Need even time extent for even-odd preconditioning"

        if IsDistributed
            @assert iseven(f.NT÷2) """
            Need time extent divisbile by 4 for even-odd preconditioning with \
            distributed fields
            """
        end
        
        return new{Backend,FloatType,IsDistributed,ArrayType,NumDirac}(f)
    end
end

function Spinorfield(
    f::SpinorfieldEO{Backend,FloatType,IsDistributed,ArrayType,NumDirac}
) where {Backend,FloatType,IsDistributed,ArrayType,NumDirac}
    return SpinorfieldEO(f.parent)
end

dims(f::SpinorfieldEO) = dims(f.parent)
local_dims(f::SpinorfieldEO) = local_dims(f.parent)
global_dims(f::SpinorfieldEO) = global_dims(f.parent)
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

Base.view(f::SpinorfieldEO, I::CartesianIndices{4}) = view(f.parent.U, I.indices...)

@inline function allindices(even::Bool, u::Union{Spinorfield,SpinorfieldEO})
    NX, NY, NZ, NT = size(u.U)
    @assert iseven(NT)
    last_range = even ? (1:div(NT, 2)) : (div(NT, 2)+1:NT)
    return CartesianIndices((NX, NY, NZ, last_range))
end

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
    update_halo!(ϕ) # TODO: Even-odd halo exchange
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

    update_halo!(ϕ) # TODO: Even-odd halo exchange
    return nothing
end

function gaussian_pseudofermions!(ϕ_eo::SpinorfieldEO{CPU,T}) where {T}
    ϕ = ϕ_eo.parent
    sz = num_dirac(ϕ) * num_colors(ϕ)
    even = true

    for e_site in eachindex(even, ϕ)
        ϕ[e_site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end

    update_halo!(ϕ) # TODO: Even-odd halo exchange
    return nothing
end

function LinearAlgebra.mul!(ϕ_eo::SpinorfieldEO{CPU,T}, α) where {T}
    ϕ = ϕ_eo.parent
    α = Complex{T}(α)
    even = true

    @batch for _site in eachindex(even, ϕ)
        ϕ[_site] *= α
    end

    update_halo!(ϕ) # TODO: Even-odd halo exchange
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

    update_halo!(ϕ) # TODO: Even-odd halo exchange
    return nothing
end

function LinearAlgebra.axpby!(
    α, ψ_eo::T, β, ϕ_eo::T, even=true
) where {T<:SpinorfieldEO{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    β = Complex{FloatT}(β)

    @batch for _site in allindices(even, ϕ)
        ϕ[_site] = α * ψ[_site] + β * ϕ[_site]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
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

    return distributed_reduce(res, +, ϕ)
end

function dot_all(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ)
        res += cdot(ϕ[site], ψ[site])
    end

    return distributed_reduce(res, +, ϕ)
end

function copy_eo!(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV
    even = true

    for e_site in allindices(even, ϕ)
        o_site = switch_sides(e_site, fdims..., NV)
        ϕ[e_site] = ψ[o_site]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    # We assume that ψ's halo is already up-to-date before calling this
    return nothing
end

function copy_oe!(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV
    odd = false

    for o_site in allindices(odd, ϕ)
        e_site = switch_sides(o_site, fdims..., NV)
        ϕ[o_site] = ψ[e_site]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    # We assume that ψ's halo is already up-to-date before calling this
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

    for e_site in allindices(even, ϕ)
        o_site = switch_sides(e_site, fdims..., NV)
        ϕ[e_site] += α * ψ[o_site]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
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

    for o_site in allindices(odd, ϕ)
        e_site = switch_sides(o_site, fdims..., NV)
        ϕ[o_site] += α * ψ[e_site]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

