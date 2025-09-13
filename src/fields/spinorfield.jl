@field_constructor Spinorfield extra_types=ND

@doc raw"""
Wrapper around a dense array or arrays of spinor/vector objects containing information about
the global MPI-topology.

    Spinorfield{B,T,ND}(NX, NY, NZ, NT)
    Spinorfield{B,T,ND}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Spinorfield(ψ::Spinorfield)
    Spinorfield(f::AbstractField; staggered=false)

Creates a Spinorfield on `B`, i.e. an array of link-variables (numcolors×ND complex vectors
with `T` precision) of size `NX × NY × NZ × NT` or a zero-initialized copy of `ψ`.
If `staggered=true`, the number of Dirac degrees of freedom (ND) is reduced to 1 instead of 4.

The data layout of the field is dependent on the backend `B`:
If `B = CPU`, then it is a 4D array of `SVector{3ND,Complex{T}}`
else, it is a 5D array of `SIMD.Vec{L,N}` with `L = {2, 4}` for `ND = {1, 4}`
with the component index being the slowest to improve coalescing on GPUs

# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Spinorfield

function Spinorfield(
    u::AbstractField{B,T,M}, ::Type{Tnew}=T; staggered=false, no_halo=false, hw=get_halo_width(u)
) where {B,T,M,Tnew}
    ND = if u isa Spinorfield || u isa SpinorfieldEO
        num_dirac(u)
    else
        staggered ? 1 : 4
    end

    u_out = if M
        ncart = get_numprocs_cart(u)
        Spinorfield{B,Tnew,ND}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Spinorfield{B,Tnew,ND}(size(u)...)
    end

    return u_out
end

@inline num_dirac(::Spinorfield{B,T,M,ND}) where {B,T,M,ND} = ND
LinearAlgebra.checksquare(f::Spinorfield) = length(f) * num_dirac(f) * num_colors(f)
Base.eltype(::Type{Spinorfield}, ::Type{T}, ::Val{ND}) where {T,ND} = SVector{3ND,Complex{T}}
# Base.eltype(::Spinorfield{CPU,T,M,ND}) where {T,M,ND} = SVector{3ND,Complex{T}}
# Base.eltype(::Spinorfield{B,T,M,1}) where {B,T,M} = SIMD.Vec{2,T}
# Base.eltype(::Spinorfield{B,T,M,4}) where {B,T,M} = SIMD.Vec{4,T}
@inline allindices(f::Spinorfield) = eachindex(f)

#### CPU Indexing ####
Base.@propagate_inbounds Base.getindex(f::Spinorfield{CPU}, i::Integer) = f.U[i]
Base.@propagate_inbounds Base.getindex(f::Spinorfield{CPU}, x, y, z, t) = f.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::Spinorfield{CPU}, site::SiteCoords) = f.U[site]
Base.@propagate_inbounds Base.setindex!(f::Spinorfield{CPU}, v, i::Integer) =
    setindex!(f.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::Spinorfield{CPU}, v, x, y, z, t) =
    setindex!(f.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::Spinorfield{CPU}, v, site::SiteCoords) =
    setindex!(f.U, v, site)
######################

#### GPU Indexing ####
Base.@propagate_inbounds function Base.getindex(
    f::Spinorfield{B,T,M,ND}, site::SiteCoords
) where {B,T,M,ND}
    return _getindex_nd(Val(ND), f.U, site, T)
end

Base.@propagate_inbounds function Base.setindex!(
    f::Spinorfield{B,T,M,ND}, v, site::SiteCoords
) where {B,T,M,ND}
    return _setindex_nd!(Val(ND), f.U, v, site, T)
end

Base.@propagate_inbounds function _getindex_nd(::Val{1}, arr, site, ::Type{T}) where T
    Base.Cartesian.@nexprs 3 i -> (
        vec = arr[i, site];
        c_i = Complex(vec[1], vec[2])
    )
    return SVector{3,Complex{T}}(c_1, c_2, c_3)
end

Base.@propagate_inbounds function _getindex_nd(::Val{4}, arr, site, ::Type{T}) where T
    Base.Cartesian.@nexprs 6 i -> (
        vec = arr[i, site];
        c_{2(i-1)+1} = Complex(vec[1], vec[2]);
        c_{2(i-1)+2} = Complex(vec[3], vec[4])
    )
    return SVector{12,Complex{T}}(
        c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12
    )
end

Base.@propagate_inbounds function _setindex_nd!(::Val{1}, arr, v, site, ::Type{T}) where {T}
    Base.Cartesian.@nexprs 3 i -> (
        arr[i, site] = SIMD.Vec{2,T}((v[i].re, v[i].im));
    )
    return nothing
end

Base.@propagate_inbounds function _setindex_nd!(::Val{4}, arr, v, site, ::Type{T}) where {T}
    Base.Cartesian.@nexprs 6 i -> (
        arr[i, site] = SIMD.Vec{4,T}((
            v[2(i-1)+1].re, v[2(i-1)+1].im,
            v[2(i-1)+2].re, v[2(i-1)+2].im));
    )
    return nothing
end
######################

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

function create_sendbuf!(ϕ::Spinorfield{B,T,M,ND}, sites, dim, dir) where {B,T,M,ND}
    ibuf = dir + 2(dim - 1)
    sendbuf = ϕ.sendbuf[ibuf]
    itr = eachindex(IndexLinear(), sites)

    parallelfor(itr, B, Val(M), Val(false), (), (), (ϕ, sendbuf)) do i, (ϕ, sendbuf)
        site = sites[i]
        setindex_buf!(sendbuf, ϕ, i, site)
    end

    return mpi_make_transferrable(sendbuf)
end

Base.@propagate_inbounds function setindex_buf!(
    sendbuf, ϕ::Spinorfield{CPU,T,M,1}, i, site
) where {T,M}
    sendbuf[i] = ϕ[site]
    return nothing
end

Base.@propagate_inbounds function setindex_buf!(
    sendbuf, ϕ::Spinorfield{CPU,T,M,4}, i, site
) where {T,M}
    sendbuf[i] = ϕ[site]
    return nothing
end

Base.@propagate_inbounds function setindex_buf!(
    sendbuf, ϕ::Spinorfield{B,T,M,1}, i, site
) where {B,T,M}
    sendbuf[1, i] = getindex_buf(ϕ, site, 1)
    sendbuf[2, i] = getindex_buf(ϕ, site, 2)
    sendbuf[3, i] = getindex_buf(ϕ, site, 3)
    return nothing
end

Base.@propagate_inbounds function setindex_buf!(
    sendbuf, ϕ::Spinorfield{B,T,M,4}, i, site
) where {B,T,M}
    sendbuf[1, i] = getindex_buf(ϕ, site, 1)
    sendbuf[2, i] = getindex_buf(ϕ, site, 2)
    sendbuf[3, i] = getindex_buf(ϕ, site, 3)
    sendbuf[4, i] = getindex_buf(ϕ, site, 4)
    sendbuf[5, i] = getindex_buf(ϕ, site, 5)
    sendbuf[6, i] = getindex_buf(ϕ, site, 6)
    return nothing
end

Base.@propagate_inbounds function getindex_buf(
    ϕ::Spinorfield{B,T,M}, site, ic
) where {B,T,M}
    return ϕ.U[ic, site]
end

function Base.copyto!(a::TF, b::TF, arange, brange) where {B,T,M,TF<:Spinorfield{B,T,M}}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    parallelfor(eachindex(IndexLinear(), arange), B, Val(M), Val(false), (), (), (a, b)) do i, (a, b)
        site_a = arange[i]
        site_b = brange[i]
        a[site_a] = b[site_b]
    end

    return nothing
end

function fill_halo!(ϕ::Spinorfield{CPU,T,M,ND}, recvbuf, siterange) where {T,M,ND}
    itr = eachindex(IndexLinear(), siterange)
    parallelfor(itr, CPU, Val(M), Val(false), (), (), (ϕ, recvbuf)) do i, (ϕ, recvbuf)
        site = siterange[i]
        ϕ[site] = recvbuf[i]
    end

    return nothing
end

function fill_halo!(ϕ::Spinorfield{B,T,M,ND}, recvbuf, siterange) where {B,T,M,ND}
    itr = eachindex(IndexLinear(), siterange)
    parallelfor(itr, B, Val(M), Val(false), (), (), (ϕ, recvbuf)) do i, (ϕ, recvbuf)
        site = siterange[i]
        ϕ[site] = _getindex_nd(Val(ND), recvbuf, i, T)
    end

    return nothing
end

function convert_field(
    ::Type{Bout}, fin::Spinorfield{CPU,Tin,M,ND}, ::Type{Tout}=Tin
) where {M,Bout,Tout,Tin,ND}
    if Bout === CPU
        fout = similar(fin, Tout)
        copy!(fout, fin)
        return fout
    end

    NX, NY, NZ, NT = size(fin)
    numprocs_cart = get_numprocs_cart(fin)
    halo_width = get_halo_width(fin)
    fout = Spinorfield{Bout,Tout,ND}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    farr = OffsetArray(array_type(Bout)(fin.U.parent), OffsetArrays.Origin(fin.U))

    parallelfor(eachindex(fout), Bout, Val(M), (fout,), (), (fout,)) do site, (fout,)
        fout[site] = farr[site]
    end

    return fout
end
