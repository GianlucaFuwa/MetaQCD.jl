@field_constructor Spinorfield extra_types=ND

@doc raw"""
Wrapper around a 4-dimensional dense array of statically sized 3xND vectors contatining
information about the global MPI-topology.

    Spinorfield{B,T,ND}(NX, NY, NZ, NT)
    Spinorfield{B,T,ND}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Spinorfield(ψ::Spinorfield)
    Spinorfield(f::AbstractField; staggered=false)

Creates a Spinorfield on `B`, i.e. an array of link-variables (numcolors×ND complex vectors
with `T` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (ND) is reduced to 1 instead of 4.
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Spinorfield

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

const MPISpinorfield{B,T,ND,AT,HT,TT} = Spinorfield{B,T,true,ND,AT,HT,TT}

@inline num_dirac(::Spinorfield{B,T,M,ND}) where {B,T,M,ND} = ND
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

function ones!(ϕ::Spinorfield{B,T,M}) where {B,T,M}
    parallelfor(eachindex(ϕ), B, Val(M), (), (ϕ,), (ϕ,)) do site, (ϕ,)
        ϕ[site] = fill(1, ϕ[site])
    end

    return nothing
end

function set_source!(ϕ::Spinorfield{B,T,M}, source::SiteCoords, a, μ) where {B,T,M}
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:NC
    vec_index = (μ - 1) * NC + a

    parallelfor(eachindex(ϕ), B, Val(M), (), (ϕ,), (ϕ,)) do site, (ϕ,)
        if site == source
            tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
            ϕ[site] = SVector{3ND,Complex{T}}(tup)
        else
            ϕ[site] = zero(SVector{3ND,Complex{T}})
        end
    end

    return nothing
end

function gaussian_pseudofermions!(ϕ::Spinorfield{B,T,M,ND}) where {B,T,M,ND}
    parallelfor(eachindex(ϕ), B, Val(M), (), (ϕ,), (ϕ,)) do site, (ϕ,)
        ϕ[site] = randn(SVector{3ND,Complex{T}}) # σ = 0.5
    end

    return nothing
end

function LinearAlgebra.mul!(ψ::TF, ϕ::TF, α) where {B,T,M,TF<:Spinorfield{B,T,M}}
    α = T(α)

    parallelfor(eachindex(ϕ, ψ), B, Val(M), (), (ψ,), (ψ, ϕ)) do site, (ψ, ϕ)
        ψ[site] = α * ϕ[site]
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ϕ::TF, ψ::TF) where {B,T,M,TF<:Spinorfield{B,T,M}}
    α = Complex{T}(α)

    parallelfor(eachindex(ϕ, ψ), B, Val(M), (), (ψ,), (ψ, ϕ)) do site, (ψ, ϕ)
        ψ[site] += α * ϕ[site]
    end

    return nothing
end

function LinearAlgebra.axpby!(α, ϕ::TF, β, ψ::TF) where {B,T,M,TF<:Spinorfield{B,T,M}}
    α = Complex{T}(α)
    β = Complex{T}(β)

    parallelfor(eachindex(ϕ, ψ), B, Val(M), (), (ψ,), (ψ, ϕ)) do site, (ψ, ϕ)
        ψ[site] = α * ϕ[site] + β * ψ[site]
    end

    return nothing
end

LinearAlgebra.norm(ϕ::Spinorfield) = sqrt(real(dot(ϕ, ϕ)))

function LinearAlgebra.dot(ϕ::TF, ψ::TF) where {B,T,M,TF<:Spinorfield{B,T,M}}
    itr = eachindex(ϕ, ψ)

    res = parallelfor_sum(itr, 0.0+0.0im, B, Val(M), (), (), (ϕ, ψ)) do d, site, (ϕ, ψ)
        d += dot(ϕ[site], ψ[site])
    end

    return distributed_reduce(res, +, ϕ)
end

function create_sendbuf!(ϕ::Spinorfield{B,T,M}, sites, dim, dir) where {B,T,M}
    ibuf = dir + 2(dim - 1)
    sendbuf = ϕ.sendbuf[ibuf]
    itr = eachindex(IndexLinear(), sites)

    parallelfor(itr, B, Val(M), (), (), (ϕ,)) do i, (ϕ,)
        site = sites[i]
        sendbuf[i] = ϕ[site]
    end

    return mpi_make_transferrable(sendbuf)[1]
end

function Base.copyto!(a::TF, b::TF, arange, brange) where {B,T,M,TF<:Spinorfield{B,T,M}}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    parallelfor(eachindex(IndexLinear(), arange), B, Val(M), (), (), (a, b)) do i, (a, b)
        site_a = arange[i]
        site_b = brange[i]
        a[site_a] = b[site_b]
    end

    return nothing
end

