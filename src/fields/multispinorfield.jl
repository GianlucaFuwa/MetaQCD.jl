@field_constructor MultiSpinorfield extra_types=ND extra_args=numspinors

@doc raw"""
Wrapper around a 5-dimensional dense array of statically 3xND vectors contatining
information about the global MPI-topology.

    MultiSpinorfield{B,T,ND}(NX, NY, NZ, NT, numspinors)
    MultiSpinorfield{B,T,ND}(NX, NY, NZ, NT, numspinors; numprocs_cart, halo_width)
    MultiSpinorfield(ψ::MultiSpinorfield)
    MultiSpinorfield(f::AbstractField; numspinors=1, staggered=false)

Creates a MultiSpinorfield on `B`, i.e. an array of link-variables (numcolors×NumDirac complex vectors
with `T` precision) of size `numspinors x NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (ND) is reduced to 1 instead of 4.
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" MultiSpinorfield

function MultiSpinorfield(
    f::MultiSpinorfield{B,T,M,ND}; no_halo=false, hw=halo_width(f)
) where {B,T,M,ND}
    global_dims = f.topology.global_dims

    u_out = if M
        ncart = f.topology.numprocs_cart
        MultiSpinorfield{B,T,ND}(
            global_dims..., f.numspinors;
            numprocs_cart=ncart, halo_width=hw, no_halo=no_halo
        )
    else
        MultiSpinorfield{B,T,ND}(global_dims..., f.numspinors)
    end

    return u_out
end

function MultiSpinorfield(
    u::AbstractField{B,T,M}, numspinors; staggered=false, no_halo=false, hw=get_halo_width(u)
) where {B,T,M}
    ND = if u isa Spinorfield
        num_dirac(u)
    else
        staggered ? 1 : 4
    end

    u_out = if M
        ncart = get_numprocs_cart(u)
        MultiSpinorfield{B,T,ND}(
            size(u)..., numspinors;
            numprocs_cart=ncart, halo_width=hw, no_halo=no_halo
        )
    else
        MultiSpinorfield{B,T,ND}(size(u)..., numspinors)
    end

    return u_out
end

const MPIMultiSpinorfield{B,T,ND,AT,HT,TT} = MultiSpinorfield{B,T,true,ND,AT,HT,TT}

# Need to overload dims and size again, because we are using 4D arrays for fermions
@inline num_dirac(::MultiSpinorfield{B,T,M,ND}) where {B,T,M,ND} = ND
@inline num_spinors(f::MultiSpinorfield) = f.numspinors
LinearAlgebra.checksquare(f::MultiSpinorfield) = length(f) * num_dirac(f) * num_colors(f)
function Base.eltype(::Type{MultiSpinorfield}, ::Type{T}, ::Val{ND}) where {T,ND}
    return SVector{3ND,Complex{T}}
end

Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, i::Integer) = f.U[i]
Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, s, x, y, z, t) = f.U[s, x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::MultiSpinorfield, s, site::SiteCoords) = f.U[s, site]

Base.@propagate_inbounds function Base.getindex(u::MPIMultiSpinorfield, is, site::SiteCoords)
    return _getindex_lat(u, is, site, u.topology.bulk_sites, u.topology.halo_width)
end

Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, i::Integer) =
    setindex!(f.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, s, x, y, z, t) =
    setindex!(f.U, v, s, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::MultiSpinorfield, v, s, site::SiteCoords) =
    setindex!(f.U, v, s, site)

Base.@propagate_inbounds function Base.setindex!(u::MPIMultiSpinorfield, v, is, site::SiteCoords)
    return _setindex_lat!(u, v, is, site, u.topology.bulk_sites, u.topology.halo_width)
end

function ones!(ϕ::MultiSpinorfield{B,T,M}) where {B,T,M}
    numspinors = ϕ.numspinors

    parallelfor(eachindex(ϕ), B, Val(M), (), (ϕ,), (ϕ,)) do site, (ϕ,)
        for is in 1:numspinors
            ϕ[is, site] = fill(1, ϕ[is, site])
        end
    end

    return nothing
end

function set_source!(ϕ::MultiSpinorfield{B,T,M}, source::SiteCoords, a, μ) where {B,T,M}
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:3
    vec_index = 3(μ - 1) + a

    parallelfor(eachindex(ϕ), B, Val(M), (), (ϕ,), (ϕ,)) do site, (ϕ,)
        if site == source
            tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
            for is in 1:ϕ.numspinors
                ϕ[is, site] = SVector{3ND,Complex{T}}(tup)
            end
        else
            for is in 1:ϕ.numspinors
                ϕ[is, site] = zero(SVector{3ND,Complex{T}})
            end
        end
    end

    return nothing
end

function gaussian_pseudofermions!(ϕ::MultiSpinorfield{B,T,M}) where {B,T,M}
    ND = num_dirac(ϕ)

    parallelfor(eachindex(ϕ), B, Val(M), (), (ϕ,), (ϕ,)) do site, (ϕ,)
        for is in 1:ϕ.numspinors
            ϕ[is, site] = randn(SVector{3ND,Complex{T}}) # σ = 0.5
        end
    end

    return nothing
end

function create_sendbuf!(ϕ::MultiSpinorfield{B,T,M}, sites, dim, dir) where {B,T,M}
    ibuf = dir + 2(dim - 1)
    sendbuf = ϕ.sendbuf[ibuf]
    itr = eachindex(IndexLinear(), sites)
    numspinors = ϕ.numspinors

    parallelfor(itr, B, Val(M), (), (), (ϕ,)) do i, (ϕ,)
        site = sites[i]

        for is in 1:numspinors
            sendbuf[is, i] = ϕ[is, site]
        end
    end

    return mpi_make_transferrable(sendbuf)[1]
end

function Base.copyto!(a::TF, b::TF, arange, brange) where {B,T,M,TF<:MultiSpinorfield{B,T,M}}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"
    @assert num_spinors(a) == num_spinors(b) "input fields must have same `numspinors`"

    parallelfor(eachindex(IndexLinear(), arange), B, Val(M), (), (), (a, b)) do i, a, b
        site_a = arange[i]
        site_b = brange[i]

        for is in 1:a.numspinors
            a[is, site_a] = b[is, site_b]
        end
    end

    return nothing
end
