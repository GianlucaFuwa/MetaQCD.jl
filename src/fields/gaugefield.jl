"""
    Gaugefield{Backend,FloatType,IsDistributed,ArrayType,GaugeAction} <: AbstractField{Backend,FloatType,IsDistributed,ArrayType}

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
struct Gaugefield{Backend,FloatType,IsDistributed,ArrayType,GaugeAction} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    
    topology::FieldTopology # Info regarding MPI topology

    β::Float64 # Seems weird to have it here, but I couldnt be bothered passing it as an argument everywhere
    Sg::Base.RefValue{Float64} # Current Gauge action, used to safe work
    CV::Base.RefValue{Float64} # Current collective variable, used to safe work
    function Gaugefield{Backend,FloatType,GaugeAction}(
        NX, NY, NZ, NT, β
    ) where {Backend,FloatType,GaugeAction}
        U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        numprocs_cart = (1, 1, 1, 1)
        halo_width = 0
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{Backend,FloatType,false,typeof(U),GaugeAction}(
            U, NX, NY, NZ, NT, NV, 3, topology, β, Sg, CV
        )
    end

    function Gaugefield{Backend,FloatType,GaugeAction}(
        NX, NY, NZ, NT, β, numprocs_cart, halo_width
    ) where {Backend,FloatType,GaugeAction}
        if prod(numprocs_cart) == 1
            return Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β)
        end

        @assert halo_width >= stencil_size(GaugeAction) "halo_width must be >= 2 when using improved gauge actions"
        NV = NX * NY * NZ * NT
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        ldims = topology.local_dims
        dims_in = ntuple(i -> ldims[i]+2halo_width, Val(4)) 
        U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, dims_in...)
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{Backend,FloatType,true,typeof(U),GaugeAction}(
            U, NX, NY, NZ, NT, NV, 3, topology, β, Sg, CV
        )
    end
end

function Gaugefield(
    u::Gaugefield{Backend,FloatType,IsDistributed,ArrayType,GaugeAction}
) where {Backend,FloatType,IsDistributed,ArrayType,GaugeAction}
    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = u.topology.halo_width
        Gaugefield{Backend,FloatType,GaugeAction}(u.NX, u.NY, u.NZ, u.NT, u.β, numprocs_cart, halo_width)
    else
        Gaugefield{Backend,FloatType,GaugeAction}(u.NX, u.NY, u.NZ, u.NT, u.β)
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

    U = if numprocs > 1
        Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β, numprocs_cart, halo_width)
    else
        Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β)
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

"""
    Colorfield{Backend,FloatType,IsDistributed,ArrayType} <: AbstractField{Backend,FloatType,IsDistributed,ArrayType}

5-dimensional dense array of statically sized 3x3 matrices contatining associated meta-data.

    Colorfield{Backend,FloatType}(NX, NY, NZ, NT)
    Colorfield{Backend,FloatType}(NX, NY, NZ, NT, numprocs_cart, halo_width)
    Colorfield(u::AbstractField)

Creates a Colorfield on `Backend`, i.e. an array of 3-by-3 `FloatType`-precision matrices of
size `4 × NX × NY × NZ × NT` or a zero-initialized Colorfield of the same size as `u`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Colorfield{Backend,FloatType,IsDistributed,ArrayType} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    
    topology::FieldTopology # Info regarding MPI topology
    function Colorfield{Backend,FloatType}(NX, NY, NZ, NT) where {Backend,FloatType}
        U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        numprocs_cart = (1, 1, 1, 1)
        halo_width = 0
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        return new{Backend,FloatType,false,typeof(U)}(U, NX, NY, NZ, NT, NV, 3, topology)
    end

    function Colorfield{Backend,FloatType}(
        NX, NY, NZ, NT, numprocs_cart, halo_width
    ) where {Backend,FloatType}
        if prod(numprocs_cart) == 1
            return Colorfield{Backend,FloatType}(NX, NY, NZ, NT)
        end

        NV = NX * NY * NZ * NT
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        ldims = topology.local_dims
        dims_in = ntuple(i -> ldims[i]+2halo_width, Val(4)) 
        U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, dims_in...)
        return new{Backend,FloatType,true,typeof(U)}(U, NX, NY, NZ, NT, NV, 3, topology)
    end
end

function Colorfield(
    u::AbstractField{Backend,FloatType,IsDistributed}
) where {Backend,FloatType,IsDistributed}
    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = u.topology.halo_width
        Colorfield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT, numprocs_cart, halo_width)
    else
        Colorfield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
end

"""
    Expfield{Backend,FloatType,IsDistributed,ArrayType} <: AbstractField{Backend,FloatType,IsDistributed,ArrayType}

5-dimensional dense array of `exp_iQ_su3` objects contatining associated meta-data. The
objects hold the `Q`-matrices and all the exponential parameters needed for stout-force
recursion.

    Expfield{Backend,FloatType}(NX, NY, NZ, NT)
    Expfield{Backend,FloatType}(NX, NY, NZ, NT, numprocs_cart, halo_width)
    Expfield(u::AbstractField)

Creates a Expfield on `Backend`, i.e. an array of `FloatType`-precison `exp_iQ_su3` objects
of size `4 × NX × NY × NZ × NT` or of the same size as `u`.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Expfield{Backend,FloatType,IsDistributed,ArrayType} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    
    topology::FieldTopology # Info regarding MPI topology
    function Expfield{Backend,FloatType}(NX, NY, NZ, NT) where {Backend,FloatType}
        U = KA.zeros(Backend(), exp_iQ_su3{FloatType}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        numprocs_cart = (1, 1, 1, 1)
        halo_width = 0
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        return new{Backend,FloatType,false,typeof(U)}(U, NX, NY, NZ, NT, NV, 3, topology)
    end

    function Expfield{Backend,FloatType}(NX, NY, NZ, NT, numprocs_cart, halo_width) where {Backend,FloatType}
        if prod(numprocs_cart) == 1
            return Expfield{Backend,FloatType}(NX, NY, NZ, NT)
        end

        NV = NX * NY * NZ * NT
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        ldims = topology.local_dims
        dims_in = ntuple(i -> ldims[i]+2halo_width, Val(4)) 
        U = KA.zeros(Backend(), exp_iQ_su3{FloatType}, 4, dims_in...)
        return new{Backend,FloatType,true,typeof(U)}(U, NX, NY, NZ, NT, NV, 3, topology)
    end
end

function Expfield(
    u::AbstractField{Backend,FloatType,IsDistributed}
) where {Backend,FloatType,IsDistributed}
    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = u.topology.halo_width
        Expfield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT, numprocs_cart, halo_width)
    else
        Expfield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
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
    elseif p == :CV
        return getfield(u, :CV)[]
    else
        return getfield(u, p)
    end
end

@inline function Base.setproperty!(u::Gaugefield, p::Symbol, val)
    if p == :Sg
        getfield(u, :Sg)[] = val
    elseif p == :CV
        getfield(u, :CV)[] = val
    else
        setproperty!(u, p, val)
    end

    return nothing
end
