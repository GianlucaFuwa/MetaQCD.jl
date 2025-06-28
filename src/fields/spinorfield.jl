"""
4-dimensional dense array of statically sized NCxND Vectors contatining associated meta-data.

    Spinorfield{B,T,ND}(NX, NY, NZ, NT)
    Spinorfield(ψ::Spinorfield)
    Spinorfield(f::AbstractField; staggered=false)

Creates a Spinorfield on `B`, i.e. an array of link-variables (numcolors×ND complex vectors
with `T` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (ND) is reduced to 1 instead of 4.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Spinorfield{B,T,M,AT,ND,HT,BT,TT} <: AbstractField{B,T,M,AT}
    U::AT # Actual field storing the gauge variables
    halos::HT
    sendbuf::BT
    topology::TT # Info regarding MPI topology
    function Spinorfield{B,T,M,ND}(
        U::AT, halos::HT, sendbuf::BT, topology::TT
    ) where {B,T,M,AT,ND,HT,BT,TT}
        check_types(B, T, U, halos, sendbuf)
        return new{B,T,M,AT,ND,HT,BT,TT}(U, halos, sendbuf, topology)
    end
end

@field_constructor Spinorfield extra_types=ND

function Spinorfield(
    u::AbstractField{B,T,M}; staggered=false, no_halo=false, hw=get_halo_width(u)
) where {B,T,M}
    ND = if u isa Spinorfield || u isa SpinorfieldEO
        num_dirac(u)
    else
        staggered ? 1 : 4
    end

    u_out = if M
        ncart = get_numprocs_cart(u)
        Spinorfield{B,T,ND}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Spinorfield{B,T,ND}(size(u)...)
    end

    return u_out
end

const MPISpinorfield{B,T,AT,ND,HT,TT} = Spinorfield{B,T,true,AT,ND,HT,TT}

@inline num_dirac(::Spinorfield{B,T,M,A,ND}) where {B,T,M,A,ND} = ND
LinearAlgebra.checksquare(f::Spinorfield) = length(f) * num_dirac(f) * num_colors(f)
function Base.eltype(::Type{Spinorfield}, ::Type{T}, ::Val{ND}) where {T,ND}
    return SVector{3ND,Complex{T}}
end

Base.@propagate_inbounds Base.getindex(f::Spinorfield, i::Integer) = f.U[i]
Base.@propagate_inbounds Base.getindex(f::Spinorfield, x, y, z, t) = f.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::Spinorfield, site::SiteCoords) = f.U[site]

Base.@propagate_inbounds function Base.getindex(u::MPISpinorfield, site::SiteCoords)
    site in u.topology.bulk_sites && return u.U[site]
    ihalo = get_halo_index(site, u.topology.bulk_sites)
    return u.halos[ihalo][site]
end

Base.@propagate_inbounds Base.setindex!(f::Spinorfield, v, i::Integer) =
    setindex!(f.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::Spinorfield, v, x, y, z, t) =
    setindex!(f.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::Spinorfield, v, site::SiteCoords) =
    setindex!(f.U, v, site)

Base.@propagate_inbounds function Base.setindex!(u::MPISpinorfield, v, site::SiteCoords)
    bulk = u.topology.bulk_sites

    if site in bulk
        u.U[site] = v
    else
        ihalo = get_halo_index(site, bulk)
        u.halos[ihalo][site] = v
    end

    return nothing
end

function ones!(ϕ::Spinorfield{B,T}) where {B,T}
    parallelfor(eachindex(ϕ), B) do site
        ϕ[site] = fill(1, ϕ[site])
    end

    return nothing
end

function set_source!(ϕ::Spinorfield{B,T}, source::SiteCoords, a, μ) where {B,T}
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:NC
    vec_index = (μ - 1) * NC + a

    parallelfor(eachindex(ϕ), B) do site
        if site == source
            tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
            ϕ[site] = SVector{3ND,Complex{T}}(tup)
        else
            ϕ[site] = zero(SVector{3ND,Complex{T}})
        end
    end

    return nothing
end

function gaussian_pseudofermions!(ϕ::Spinorfield{B,T,M,AT,ND}) where {B,T,M,AT,ND}
    parallelfor(eachindex(ϕ), B) do site
        ϕ[site] = randn(SVector{3ND,Complex{T}}) # σ = 0.5
    end

    return nothing
end

function LinearAlgebra.mul!(ψ::TF, ϕ::TF, α) where {B,T,TF<:Spinorfield{B,T}}
    α = T(α)

    parallelfor(eachindex(ϕ, ψ), B) do site
        ψ[site] = α * ϕ[site]
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ψ::TF, ϕ::TF) where {B,T,TF<:Spinorfield{B,T}}
    α = Complex{T}(α)

    parallelfor(eachindex(ϕ, ψ), B) do site
        ϕ[site] += α * ψ[site]
    end

    return nothing
end

function LinearAlgebra.axpby!(α, ψ::TF, β, ϕ::TF) where {B,T,TF<:Spinorfield{B,T}}
    α = Complex{T}(α)
    β = Complex{T}(β)

    parallelfor(eachindex(ϕ, ψ), B) do site
        ϕ[site] = α * ψ[site] + β * ϕ[site]
    end

    return nothing
end

LinearAlgebra.norm(ϕ::Spinorfield) = sqrt(real(dot(ϕ, ϕ)))

function LinearAlgebra.dot(ϕ::T, ψ::T) where {B,T<:Spinorfield{B}}
    res = parallelfor_sum(eachindex(ϕ, ψ), 0.0+0.0im, B) do d, site
        d += dot(ϕ[site], ψ[site])
    end

    return distributed_reduce(res, +, ϕ)
end

function create_sendbuf!(f::Spinorfield{B}, sites, dim, dir) where {B}
    ibuf = dir + 2(dim - 1)
    sendbuf = f.sendbuf[ibuf]

    parallelfor(eachindex(IndexLinear(), sites), B) do i
        site = sites[i]
        sendbuf[i] = f[site]
    end

    return sendbuf
end

function Base.copyto!(a::T, b::T, arange, brange) where {B,T<:Spinorfield{B}}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    parallelfor(eachindex(IndexLinear(), arange), B) do i
        site_a = arange[i]
        site_b = brange[i]
        a[site_a] = b[site_b]
    end

    return nothing
end

