"""
5-dimensional dense array of statically sized 3x3 matrices contatining associated meta-data.

    Gaugefield{B,T,GA}(NX, NY, NZ, NT, β)
    Gaugefield{B,T,GA}(NX, NY, NZ, NT, β, numprocs_cart, halo_width)
    Gaugefield(U::Gaugefield)
    Gaugefield(parameters::ParameterSet)

Creates a Gaugefield on backend `B`, i.e. an array of link-variables (SU3 matrices with
`T` precision) of size `4 × NX × NY × NZ × NT` with coupling parameter `β` and gauge
action `GA` or a zero-initialized copy of `U`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
# Supported gauge actions
`WilsonGaugeAction` \\
`SymanzikTreeGaugeAction` (Lüscher-Weisz) \\
`IwasakiGaugeAction` \\
`DBW2GaugeAction`
"""
struct Gaugefield{B,T,M,AT,GA,HT,BT,TT} <: AbstractField{B,T,M,AT}
    U::AT # Actual field storing the gauge variables
    halos::HT
    sendbuf::BT
    topology::TT # Info regarding MPI topology
    β::Float64 # Seems weird to have it here, but I couldnt be bothered passing it as an argument everywhere
    function Gaugefield{B,T,M,GA}(
        U::AT, halos::HT, sendbuf::BT, topology::TT, β
    ) where {B,T,M,AT,GA,HT,BT,TT}
        check_types(B, T, U, halos, sendbuf)
        return new{B,T,M,AT,GA,HT,BT,TT}(U, halos, sendbuf, topology, β)
    end
end

@field_constructor Gaugefield extra_types=GA extra_args=β

function Gaugefield(
    u::Gaugefield{B,T,M,AT,GA}; no_halo=false, hw=get_halo_width(u)
) where {B,T,M,AT,GA}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Gaugefield{B,T,GA}(
            size(u)..., u.β, numprocs_cart=ncart, halo_width=hw, no_halo=no_halo
        )
    else
        Gaugefield{B,T,GA}(size(u)..., u.β)
    end

    return u_out
end

function Gaugefield(parameters)
    global_dims = parameters.L
    β = parameters.beta
    GA = GAUGE_ACTION[parameters.gauge_action]
    T = Utils.FLOAT_TYPE[parameters.float_type]
    B = BACKENDS[parameters.backend]
    ncart = parameters.numprocs_cart
    hw = parameters.halo_width

    U = Gaugefield{B,T,GA}(global_dims..., β, numprocs_cart=ncart, halo_width=hw)

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

@inline gauge_action(::Gaugefield{B,T,M,AT,GA}) where {B,T,M,AT,GA} = GA
Base.eltype(::Type{Gaugefield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}
# Base.eltype(::Gaugefield{B,T}) where {B,T} = SMatrix{3,3,Complex{T},9}

function create_sendbuf!(u::AbstractField{B}, sites, dim, dir) where {B}
    ibuf = dir + 2(dim - 1)
    sendbuf = u.sendbuf[ibuf]

    parallelfor(eachindex(IndexLinear(), sites), B) do i
        site = sites[i]

        for μ in 1:4
            sendbuf[μ, i] = u[μ, site]
        end
    end

    return sendbuf
end

function Base.copyto!(a::T, b::T, arange, brange) where {B,T<:AbstractField{B}}
    @assert length(arange) == length(brange) "send buffer and recv buffer arent of same size"

    parallelfor(eachindex(IndexLinear(), arange), B) do i
        site_a = arange[i]
        site_b = brange[i]

        for μ in 1:4
            a[μ, site_a] = b[μ, site_b]
        end
    end

    return nothing
end
