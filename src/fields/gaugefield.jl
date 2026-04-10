@field_constructor Gaugefield extra_types=GA,N extra_args=β

@doc raw"""
Wrapper around a dense array or arrays of SU3 objects containing information about
the global MPI-topology.

    Gaugefield{B,T,GA,NFLOAT}(NX, NY, NZ, NT, β)
    Gaugefield{B,T,GA,NFLOAT}(NX, NY, NZ, NT, β; numprocs_cart, halo_width)
    Gaugefield(U::Gaugefield; no_halo, hw)
    Gaugefield(parameters::ParameterSet)

Creates a Gaugefield on backend `B`, i.e. an array of link-variables (SU3 matrices with
`T` precision) of size `4 × NX × NY × NZ × NT` with coupling parameter `β` and gauge
action `GA` or a zero-initialized copy of `U`.
`NFLOAT` specifies the number of floating point numbers per SU3 element (12, or 18) but for
now it is bound to 18 on CPU backends.

The data layout of the field is dependent on the backend `B`:
If `B = CPU`, then it is a 5D array of `SMatrix{3,3,Complex{T},9}`
else, it is a 4-tuple of 5D arrays of `SIMD.Vec{L,N}` with `L = {4, 2}` for `NFLOAT = {12, 18}`
with the component index being the slowest to improve coalescing on GPUs

# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
# Supported gauge actions
`WilsonGaugeAction` \
`SymanzikTreeGaugeAction` (Lüscher-Weisz) \
`IwasakiGaugeAction` \
`DBW2GaugeAction`
""" Gaugefield

function Gaugefield(
    u::Gaugefield{B,T,M,GA,N}, ::Type{Tnew}=T; no_halo=false, halo_width=get_halo_width(u)
) where {B,T,M,GA,N,Tnew}
    u_out = if M
        numprocs_cart = get_numprocs_cart(u)
        Gaugefield{B,Tnew,GA,N}(size(u)..., u.β; numprocs_cart, halo_width, no_halo)
    else
        Gaugefield{B,Tnew,GA,N}(size(u)..., u.β)
    end

    return u_out
end

function Gaugefield(parameters)
    global_dims = parameters.L
    β = parameters.beta
    GA = GAUGE_ACTION[parameters.gauge_action]
    N = parameters.su3_nfloats
    T = Utils.FLOAT_TYPE[parameters.float_type]
    B = BACKENDS[parameters.backend]
    numprocs_cart = parameters.numprocs_cart
    halo_width = parameters.halo_width
    B == CPU && @assert(N == 18, "su3_nfloats is bound to 18 on CPUs for now")

    U = Gaugefield{B,T,GA,N}(global_dims..., β; numprocs_cart, halo_width)

    initial = parameters.initial
    if initial == "cold"
        identity_gauges!(U)
    elseif initial == "hot"
        random_gauges!(U)
    else
        error("intial condition \"$(initial)\" not supported, only \"cold\" or \"hot\"")
    end

    return U
end

@inline num_floats(::Gaugefield{B,T,M,GA,N}) where {B,T,M,GA,N} = N

#### CPU Indexing ####
@inline function add_directional_indices(::AbstractField{CPU}, siterange::CartesianIndices)
    return CartesianIndices((4, siterange.indices...))
end

@inline function add_all_indices(::AbstractField{CPU}, siterange::CartesianIndices)
    return CartesianIndices((4, siterange.indices...))
end

@inline function add_all_indices(::Gaugefield{CPU,T,M,GA,12}, siterange::CartesianIndices) where {T,M,GA}
    return CartesianIndices((4, siterange.indices...))
end

Base.@propagate_inbounds Base.getindex(u::Gaugefield{CPU}, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.getindex(u::Gaugefield{CPU}, μsite) = u.U[μsite]
Base.@propagate_inbounds Base.setindex!(u::Gaugefield{CPU}, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)
Base.@propagate_inbounds Base.setindex!(u::Gaugefield{CPU}, v, μsite) =
    setindex!(u.U, v, μsite)
######################

#### GPU Indexing ####
@inline function add_directional_indices(::AbstractField{B}, siterange::CartesianIndices) where {B}
    return CartesianIndices((siterange.indices..., 4))
end

@inline function add_all_indices(::AbstractField{B}, siterange::CartesianIndices) where {B}
    return CartesianIndices((9, siterange.indices..., 4))
end

@inline function add_all_indices(::Gaugefield{B,T,M,GA,12}, siterange::CartesianIndices) where {B,T,M,GA}
    return CartesianIndices((3, siterange.indices..., 4))
end

Base.@propagate_inbounds function Base.getindex(
    u::Gaugefield{B,T}, ii::Integer
) where {B,T}
    return u.U[ii]
end

Base.@propagate_inbounds function Base.getindex(
    u::Gaugefield{B,T,M,GA,N}, μ, site::SiteCoords
) where {B,T,M,GA,N}
    return _getindex_mat(Val(N), u.U, μ, site, T)
end

Base.@propagate_inbounds function Base.getindex(
    u::Gaugefield{B,T,M,GA,N}, μsite
) where {B,T,M,GA,N}
    return _getindex_mat(Val(N), u.U, μsite, T)
end

Base.@propagate_inbounds function Base.setindex!(
    u::Gaugefield{B,T}, v, ii::Integer
) where {B,T}
    u.U[ii] = v
    return nothing
end

Base.@propagate_inbounds function Base.setindex!(
    u::Gaugefield{B,T,M,GA,N}, v, μ, site::SiteCoords
) where {B,T,M,GA,N}
    return _setindex_mat!(Val(N), u.U, v, μ, site, T)
end

Base.@propagate_inbounds function Base.setindex!(
    u::Gaugefield{B,T,M,GA,N}, v, μsite
) where {B,T,M,GA,N}
    return _setindex_mat!(Val(N), u.U, v, μsite, T)
end

Base.@propagate_inbounds function _getindex_mat(
    ::Val{12}, arr, μ, site, ::Type{T}
) where {T}
    Base.Cartesian.@nexprs 3 i -> (
        vec = arr[i, site, μ];
        c1_i = Complex(vec[1], vec[2]);
        c2_i = Complex(vec[3], vec[4]);
    )
    return reconstruct_su3(c1_1, c2_1, c1_2, c2_2, c1_3, c2_3)
end

Base.@propagate_inbounds function _getindex_mat(
    ::Val{18}, arr, μ, site, ::Type{T}
) where {T}
    tup = ntuple(Val(9)) do i
        vec = arr[i, site, μ];
        Complex(vec[1], vec[2])
    end
    return SMatrix{3,3,Complex{T},9}(tup)
end

Base.@propagate_inbounds function _getindex_mat(
    ::Val{12}, arr, ii, ::Type{T}
) where {T}
    Base.Cartesian.@nexprs 3 i -> (
        vec = arr[i, ii];
        c1_i = Complex(vec[1], vec[2]);
        c2_i = Complex(vec[3], vec[4]);
    )
    return reconstruct_su3(c1_1, c2_1, c1_2, c2_2, c1_3, c2_3)
end

Base.@propagate_inbounds function _getindex_mat(
    ::Val{18}, arr, ii, ::Type{T}
) where {T}
    tup = ntuple(Val(9)) do i
        vec = arr[i, ii];
        Complex(vec[1], vec[2])
    end
    return SMatrix{3,3,Complex{T},9}(tup)
end

Base.@propagate_inbounds function _setindex_mat!(
    ::Val{12}, arr, v, μ, site, ::Type{T}
) where {T}
    Base.Cartesian.@nexprs 3 i -> (
        arr[i, site, μ] = SIMD.Vec{4,T}((
            v[2(i-1)+1].re, v[2(i-1)+1].im, v[2(i-1)+2].re, v[2(i-1)+2].im));
    )
    return nothing
end

Base.@propagate_inbounds function _setindex_mat!(
    ::Val{18}, arr, v, μ, site, ::Type{T}
) where {T}
    Base.Cartesian.@nexprs 9 i -> (
        arr[i, site, μ] = SIMD.Vec{2,T}((v[i].re, v[i].im));
    )
    return nothing
end

Base.@propagate_inbounds function _setindex_mat!(
    ::Val{12}, arr, v, ii, ::Type{T}
) where {T}
    Base.Cartesian.@nexprs 3 i -> (
        arr[i, ii] = SIMD.Vec{4,T}((
            v[2(i-1)+1].re, v[2(i-1)+1].im, v[2(i-1)+2].re, v[2(i-1)+2].im));
    )
    return nothing
end

Base.@propagate_inbounds function _setindex_mat!(
    ::Val{18}, arr, v, ii, ::Type{T}
) where {T}
    Base.Cartesian.@nexprs 9 i -> (
        arr[i, ii] = SIMD.Vec{2,T}((v[i].re, v[i].im));
    )
    return nothing
end
######################

@inline gauge_action(::Gaugefield{B,T,M,GA}) where {B,T,M,GA} = GA
@inline nfloat(::Gaugefield{B,T,M,GA,N}) where {B,T,M,GA,N} = N
Base.eltype(::Type{Gaugefield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}
# Base.eltype(::Gaugefield{CPU,T,M,GA,N}) where {T,M,GA,N} = SMatrix{3,3,Complex{T},9}
# Base.eltype(::Gaugefield{B,T,M,GA,12}) where {B,T,M,GA} = SIMD.Vec{4,T}
# Base.eltype(::Gaugefield{B,T,M,GA,18}) where {B,T,M,GA} = SIMD.Vec{2,T}

function convert_field(
    ::Type{Bout}, Uin::Gaugefield{CPU,Tin,M,GA,N}, ::Type{Tout}=Tin
) where {M,Bout,Tout,Tin,GA,N}
    if Bout === CPU
        Uout = similar(Uin, Tout)
        copy!(Uout, Uin)
        return Uout
    end

    NX, NY, NZ, NT = size(Uin)
    numprocs_cart = get_numprocs_cart(Uin)
    halo_width = get_halo_width(Uin)
    Uout = Gaugefield{Bout,Tout,GA,N}(NX, NY, NZ, NT, Uin.β; numprocs_cart, halo_width)
    Uarr = OffsetArray(array_type(Bout)(Uin.U.parent), OffsetArrays.Origin(Uin.U))

    parallelfor(eachindex(Uout), Bout, Val(M), (Uout,), (), (Uout,)) do site, (Uout,)
        Uout[1, site] = Uarr[1, site]
        Uout[2, site] = Uarr[2, site] 
        Uout[3, site] = Uarr[3, site] 
        Uout[4, site] = Uarr[4, site] 
    end

    return Uout
end

function convert_field(
    ::Type{Bout}, Uin::Gaugefield{Bin,Tin,M,GA,N}, ::Type{Tout}=Tin
) where {M,Bout,Tout,Bin,Tin,GA,N}
    if Bout === Bin
        Uout = similar(Uin, Tout)
        copy!(Uout, Uin)
        return Uout
    end

    NX, NY, NZ, NT = size(Uin)
    numprocs_cart = get_numprocs_cart(Uin)
    halo_width = get_halo_width(Uin)
    Uout = Gaugefield{Bout,Tout,GA,N}(NX, NY, NZ, NT, Uin.β; numprocs_cart, halo_width)
    Uarr = OffsetArray(array_type(Bout)(Uin.U.parent), OffsetArrays.Origin(Uin.U))

    parallelfor(eachindex(Uout), Bout, Val(M), (Uout,), (), (Uout,)) do site, (Uout,)
        Uout[1, site] = _getindex_mat(Val(N), Uarr, 1, site, Tout)
        Uout[2, site] = _getindex_mat(Val(N), Uarr, 2, site, Tout)
        Uout[3, site] = _getindex_mat(Val(N), Uarr, 3, site, Tout)
        Uout[4, site] = _getindex_mat(Val(N), Uarr, 4, site, Tout)
    end

    return Uout
end
