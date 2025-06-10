"""
5-dimensional dense array of statically sized 3x3 matrices contatining associated meta-data.

    Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β)
    Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β, numprocs_cart, halo_width)
    Gaugefield(U::Gaugefield)
    Gaugefield(parameters::ParameterSet)

Creates a Gaugefield on `Backend`, i.e. an array of link-variables (SU3 matrices with
`FloatType` precision) of size `4 × NX × NY × NZ × NT` with coupling parameter `β` and gauge
action `GaugeAction` or a zero-initialized copy of `U`
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
struct Gaugefield{Backend,FloatType,IsDistributed,ArrayType,GaugeAction,BufferType} <:
       AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    send_buf::BufferType
    recv_buf::BufferType
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    topology::FieldTopology # Info regarding MPI topology

    β::Float64 # Seems weird to have it here, but I couldnt be bothered passing it as an argument everywhere
    Sg::Base.RefValue{Float64} # Current Gauge action, used to safe work
    CV::Vector{Float64} # Current collective variable, used to safe work
    function Gaugefield{B,T,M,AT,GA,BT}(
        U::AT, send_buf::BT, recv_buf::BT, NX, NY, NZ, NT, NV, NC, topology, β, Sg, CV
    ) where {B,T,M,AT,GA,BT}
        # some sanity checks
        @assert get_backend(U) isa Backend
        if BT !== Nothing
            @assert eltype(U) == eltype(send_buf) == eltype(recv_buf)
        end
        @assert eltype(eltype(U)) === Complex{T}
        return new{B,T,M,AT,GA,BT}(
            U, send_buf, recv_buf, NX, NY, NZ, NT, NV, NC, topology, β, Sg, CV
        )
    end
end

function Gaugefield{Backend,FloatType,GaugeAction}(
    NX, NY, NZ, NT, β; ncv=1
) where {Backend,FloatType,GaugeAction}
    U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, NX, NY, NZ, NT)
    send_buf = recv_buf = nothing
    NV = NX * NY * NZ * NT
    numprocs_cart = (1, 1, 1, 1)
    halo_width = 0
    topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
    Sg = Base.RefValue{Float64}(0.0)
    CV = zeros(Float64, ncv)
    return Gaugefield{Backend,FloatType,false,typeof(U),GaugeAction,Nothing}(
        U, send_buf, recv_buf, NX, NY, NZ, NT, NV, 3, topology, β, Sg, CV
    )
end

function Gaugefield{Backend,FloatType,GaugeAction}(
    NX, NY, NZ, NT, β, numprocs_cart, halo_width; ncv=1
) where {Backend,FloatType,GaugeAction}
    if prod(numprocs_cart) == 1
        return Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β)
    end

    @assert halo_width >= stencil_size(GaugeAction) """
    halo_width must be >= 2 when using improved gauge actions
    """

    NV = NX * NY * NZ * NT
    topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
    ldims = topology.local_dims_padded

    halo_width = topology.halo_width
    origin = OffsetArrays.Origin((1, topology.bulk_sites[1].I .- halo_width...)...)
    U = OffsetArray(KA.zeros(Backend(), SU{3,9,FloatType}, 4, ldims...), origin)

    buf_length = 4maximum(sz for sz in topology.halo_sizes)
    send_buf = KA.zeros(Backend(), SU{3,9,FloatType}, buf_length)
    recv_buf = KA.zeros(Backend(), SU{3,9,FloatType}, buf_length)

    for irank in 0:mpi_size()-1
        if mpi_myrank() == irank
            @show topology.bulk_sites
            @show shrink_bulk(topology.bulk_sites, 1)
            @show topology.border_sites[2]
            println()
        end
        mpi_barrier()
    end

    Sg = Base.RefValue{Float64}(0.0)
    CV = zeros(Float64, ncv)
    return Gaugefield{Backend,FloatType,true,typeof(U),GaugeAction,typeof(send_buf)}(
        U, send_buf, recv_buf, NX, NY, NZ, NT, NV, 3, topology, β, Sg, CV
    )
end

function Gaugefield(
    u::Gaugefield{Backend,FloatType,IsDistributed,ArrayType,GaugeAction}
) where {Backend,FloatType,IsDistributed,ArrayType,GaugeAction}
    ncv = length(u.CV)

    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = maximum(u.topology.halo_width)
        Gaugefield{Backend,FloatType,GaugeAction}(
            u.NX, u.NY, u.NZ, u.NT, u.β, numprocs_cart, halo_width; ncv=ncv
        )
    else
        Gaugefield{Backend,FloatType,GaugeAction}(u.NX, u.NY, u.NZ, u.NT, u.β; ncv=ncv)
    end

    return u_out
end

function Gaugefield(parameters)
    NX, NY, NZ, NT = parameters.L
    β = parameters.beta
    GaugeAction = GAUGE_ACTION[parameters.gauge_action]
    FloatType = Utils.FLOAT_TYPE[parameters.float_type]
    Backend = BACKENDS[parameters.backend]
    numprocs_cart = parameters.numprocs_cart
    numprocs = sum(numprocs_cart)
    halo_width = parameters.halo_width
    ncv = length(parameters.biases)

    U = if numprocs > 1
        Gaugefield{Backend,FloatType,GaugeAction}(
            NX, NY, NZ, NT, β, numprocs_cart, halo_width; ncv=ncv
        )
    else
        Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β; ncv=ncv)
    end

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

@inline function gauge_action(
    ::Gaugefield{Backend,FloatType,IsDistributed,ArrayType,GaugeAction}
) where {Backend,FloatType,IsDistributed,ArrayType,GaugeAction}
    return GaugeAction
end

# overload getproperty and setproperty! for convenience
@inline function Base.getproperty(u::Gaugefield, p::Symbol)
    if p == :Sg
        return getfield(u, :Sg)[]
    else
        return getfield(u, p)
    end
end

@inline function Base.setproperty!(u::Gaugefield, p::Symbol, val)
    if p == :Sg
        getfield(u, :Sg)[] = val
    elseif p == :CV
        getfield(u, :CV) .= val
    else
        setfield!(u, p, val)
    end

    return nothing
end
