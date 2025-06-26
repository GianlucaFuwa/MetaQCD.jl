struct SpinorfieldEO{B,T,M,AT,ND} <: AbstractField{B,T,M,AT}
    parent::Spinorfield{B,T,M,AT,ND}
    function SpinorfieldEO(
        f::Spinorfield{B,T,M,AT,ND}
    ) where {B,T,M,AT,ND}
        @assert iseven(size(f, 4)) "Need even time extent for even-odd preconditioning"

        if M
            @assert iseven(size(f, 4)÷2) """
            Need time extent divisbile by 4 for even-odd preconditioning with \
            distributed fields
            """
        end
        
        return new{B,T,M,AT,ND}(f)
    end
end

"""
    even_odd(f::Spinorfield)

Create a wrapper around a `Spinorfield` to signal that it is meant to be used in the context
of even-odd preconditioning. What this amounts to is that we realign the entries such that
`ϕ -> (ϕₑ, ϕₒ)`, which is achieved by recalculating the index whenever we index into `ϕ`
or iterating only over one half of its indices.
"""
even_odd(f::Spinorfield) = SpinorfieldEO(f)
even_odd(f::SpinorfieldEO) = f

const MPISpinorfieldEO{B,T,AT,ND} = SpinorfieldEO{B,T,true,AT,ND}

@inline num_dirac(::SpinorfieldEO{B,T,M,A,ND}) where {B,T,M,A,ND} = ND
LinearAlgebra.checksquare(f::SpinorfieldEO) = length(f) * num_dirac(f) * num_colors(f)
function Base.eltype(::Type{SpinorfieldEO}, ::Type{T}, ::Val{ND}) where {T,ND}
    return SVector{3ND,Complex{T}}
end

KA.get_backend(f::SpinorfieldEO) = get_backend(f.parent)
Base.length(f::SpinorfieldEO) = length(f.parent)
Base.size(f::SpinorfieldEO) = size(f.parent)
Base.size(f::SpinorfieldEO, μ) = size(f.parent, μ)
Base.axes(f::SpinorfieldEO) = axes(f.parent)
Base.axes(f::SpinorfieldEO, μ::Integer) = axes(f.parent, μ)
@inline get_local_dims(f::SpinorfieldEO) = get_local_dims(f.parent)
@inline get_global_dims(f::SpinorfieldEO) = get_global_dims(f.parent)
@inline get_local_volume(f::SpinorfieldEO) = get_local_volume(f.parent)
@inline get_global_volume(f::SpinorfieldEO) = get_global_volume(f.parent)
@inline get_halo_width(f::SpinorfieldEO) = get_halo_width(f.parent)
@inline get_numprocs_cart(f::SpinorfieldEO) = get_numprocs_cart(f.parent)

Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, i::Integer) = f.parent.U[i]
Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, x, y, z, t) = f.parent.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, site::SiteCoords) = f.parent.U[site]
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, i::Integer) =
    setindex!(f.parent.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, x, y, z, t) =
    setindex!(f.parent.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, site::SiteCoords) =
    setindex!(f.parent.U, v, site)

function Base.getproperty(f::SpinorfieldEO, name::Symbol)
    if name == :parent
        return getfield(f, name)
    else
        return getfield(f.parent, name)
    end
end

@inline function allindices(even::Bool, u::Union{Spinorfield,SpinorfieldEO})
    _, NX, NY, NZ, NT = size(u.U)
    @assert iseven(NT)
    last_range = even ? (1:div(NT, 2)) : (div(NT, 2)+1:NT)
    return CartesianIndices((NX, NY, NZ, last_range))
end

clear!(ϕ_eo::SpinorfieldEO) = clear!(ϕ_eo.parent)
ones!(ϕ_eo::SpinorfieldEO) = ones!(ϕ_eo.parent)

function Base.copy!(ϕ_eo::T, ψ_eo::T) where {B,T<:SpinorfieldEO{B}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even_half = true

    parallelfor(eachindex(even_half, ϕ, ψ), B) do e_site
        ϕ[e_site] = ψ[e_site]
    end

    return nothing
end

function set_source!(ϕ_eo::SpinorfieldEO{B,T}, source::SiteCoords, a, μ) where {B,T}
    ϕ = ϕ_eo.parent
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:3
    vec_index = (μ - 1) * NC + a

    parallelfor(eachindex(ϕ), B) do site
        if site == source
            tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
            _site = map_to_half(site, eachindex(ϕ))
            ϕ[_site] = SVector{3ND,Complex{T}}(tup)
        else
            ϕ[_site] = zero(SVector{3ND,Complex{T}})
        end
    end
    return nothing
end

function gaussian_pseudofermions!(ϕ_eo::SpinorfieldEO{B,T,M,AT,ND}) where {B,T,M,AT,ND}
    ϕ = ϕ_eo.parent
    even_half = true

    parallelfor(eachindex(even_half, ϕ), B) do e_site
        ϕ[e_site] = randn(SVector{3ND,Complex{T}}) # σ = 0.5
    end

    return nothing
end

function LinearAlgebra.mul!(ψ_eo::TF, ϕ_eo::TF, α) where {B,T,TF<:SpinorfieldEO{B,T}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    α = Complex{T}(α)
    even_half = true

    parallelfor(eachindex(even_half, ϕ, ψ), B) do e_site
        ψ[e_site] = ϕ[e_site] * α
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ψ_eo::TF, ϕ_eo::TF) where {B,T,TF<:SpinorfieldEO{B,T}} # even on even is the default
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    α = Complex{T}(α)
    even_half = true

    parallelfor(eachindex(even_half, ϕ, ψ), B) do e_site
        ϕ[e_site] += α * ψ[e_site]
    end

    return nothing
end

function LinearAlgebra.axpby!(
    α, ψ_eo::TF, β, ϕ_eo::TF, even_half=true
) where {B,T,TF<:SpinorfieldEO{B,T}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    α = Complex{T}(α)
    β = Complex{T}(β)

    parallelfor(eachindex(even_half, ϕ, ψ), B) do _site
        ϕ[_site] = α * ψ[_site] + β * ϕ[_site]
    end

    return nothing
end

LinearAlgebra.norm(ϕ_eo::SpinorfieldEO) = sqrt(real(dot(ϕ_eo, ϕ_eo)))

function LinearAlgebra.dot(ϕ_eo::T, ψ_eo::T) where {B,T<:SpinorfieldEO{B}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision
    even_half = true

    res = parallelfor_sum(eachindex(even_half, ϕ, ψ), 0.0+0.0im, B) do d, e_site
        d += dot(ϕ[e_site], ψ[e_site])
    end

    return distributed_reduce(res, +, ϕ)
end

function create_sendbuf!(f::SpinorfieldEO, sites, dim, dir)
    return create_sendbuf!(f.parent, sites, dim, dir)
end

function Base.copyto!(a::T, b::T, arange, brange) where {T<:SpinorfieldEO}
    # XXX: only do even sites?
    return copyto!(a.parent, b.parent, arange, brange)
end

