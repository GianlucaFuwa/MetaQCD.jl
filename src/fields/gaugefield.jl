"""
    Gaugefield{Backend,FloatType,MPIParallel,ArrayType,HaloType,GaugeAction} <: AbstractField{Backend,FloatType,MPIParallel,ArrayType}

5-dimensional dense array of statically sized 3x3 matrices contatining associated meta-data.

    Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β)
    Gaugefield{Backend,FloatType,GaugeAction}(NX, NY, NZ, NT, β, nprocs_cart, pad)
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
struct Gaugefield{B,T,M,A,H,GA} <: AbstractField{B,T,M,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    halo_sendbuf::H
    halo_recvbuf::H
    pad::Int64 # halo width
    
    # Meta-data regarding portion of the field local to current process
    my_NX::Int64
    my_NY::Int64
    my_NZ::Int64
    my_NT::Int64
    my_NV::Int64
    # XXX:
    # pack these into a MPITopology or TopologyInfo struct
    nprocs::Int64
    nprocs_cart::NTuple{4,Int64}
    myrank::Int64
    myrank_cart::NTuple{4,Int64}
    comm_cart::Utils.Comm

    β::Float64 # Seems weird to have it here, but I couldnt be bothered passing it as an argument everywhere
    Sg::Base.RefValue{Float64} # Current Gauge action, used to safe work
    CV::Base.RefValue{Float64} # Current collective variable, used to safe work
    function Gaugefield{B,T,GA}(NX, NY, NZ, NT, β) where {B,T,GA}
        U = KA.zeros(B(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        halo_sendbuf = halo_recvbuf = nothing
        pad = 0
        nprocs = 1
        nprocs_cart = (1, 1, 1, 1)
        myrank_cart = (0, 0, 0, 0)
        comm_cart = mpi_cart_create(nprocs_cart; periodic=map(_->true, nprocs_cart))
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{B,T,false,typeof(U),typeof(halo_sendbuf),GA}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, NX, NY, NZ, NT, NV,
            nprocs, nprocs_cart, mpi_myrank(), myrank_cart, comm_cart, β, Sg, CV,
        )
    end

    function Gaugefield{B,T,GA}(NX, NY, NZ, NT, β, nprocs_cart, pad) where {B,T,GA}
        sum(nprocs_cart) == 1 && return Gaugefield{B,T,GA}(NX, NY, NZ, NT, β)
        NV = NX * NY * NZ * NT
        nX, nY, nZ, nT = nprocs_cart
        nprocs = nX * nY * nZ * nT
        @assert (NX%nX, NY%nY, NZ%nZ, NT%nT) == (0, 0, 0, 0) "Lattice dimensions must be divisible by comm dimensions"
        my_NX, my_NY, my_NZ, my_NT = (NX÷nX, NY÷nY, NZ÷nZ, NT÷nT)
        my_NV = my_NX * my_NY * my_NZ * my_NT
        comm_cart = mpi_cart_create(nprocs_cart; periodic=map(_->true, nprocs_cart))
        myrank_cart = (mpi_cart_coords(comm_cart, mpi_myrank())...,)
        @assert pad >= stencil_size(GA) "halo_width must be >= 2 when using improved gauge actions"
        max_halo_dims = maximum([prod(halo_dims((NX, NY, NZ, NT), pad, i)) for i in 1:4])
        halo_sendbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        halo_recvbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        U = KA.zeros(B(), SU{3,9,T}, 4, my_NX+2pad, my_NY+2pad, my_NZ+2pad, my_NT+2pad)
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{B,T,true,typeof(U),typeof(halo_sendbuf),GA}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, my_NX, my_NY, my_NZ,
            my_NT, my_NV, nprocs, nprocs_cart, mpi_myrank(), myrank_cart, comm_cart, β, Sg, CV,
        )
    end
end

function Gaugefield(u::Gaugefield{B,T,M,A,GA}) where {B,T,M,A,GA}
    u_out = if M
        Gaugefield{B,T,GA}(u.NX, u.NY, u.NZ, u.NT, u.β, u.nprocs_cart, u.pad)
    else
        Gaugefield{B,T,GA}(u.NX, u.NY, u.NZ, u.NT, u.β)
    end

    return u_out
end

function Gaugefield(parameters)
    NX, NY, NZ, NT = parameters.L
    β = parameters.beta
    GA = GAUGE_ACTION[parameters.gauge_action]
    T = Utils.FLOAT_TYPE[parameters.float_type]
    B = BACKENDS[parameters.backend]
    nprocs_cart = parameters.nprocs_cart
    nprocs = sum(nprocs_cart)
    pad = parameters.halo_width

    U = if nprocs > 1
        Gaugefield{B,T,GA}(NX, NY, NZ, NT, β, nprocs_cart, pad)
    else
        Gaugefield{B,T,GA}(NX, NY, NZ, NT, β)
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
    Colorfield{Backend,FloatType,MPIParallel,ArrayType,HaloType} <: AbstractField{Backend,FloatType,MPIParallel,ArrayType}

5-dimensional dense array of statically sized 3x3 matrices contatining associated meta-data.

    Colorfield{Backend,FloatType}(NX, NY, NZ, NT)
    Colorfield{Backend,FloatType}(NX, NY, NZ, NT, nprocs_cart, pad)
    Colorfield(u::AbstractField)

Creates a Colorfield on `Backend`, i.e. an array of 3-by-3 `FloatType`-precision matrices of
size `4 × NX × NY × NZ × NT` or a zero-initialized Colorfield of the same size as `u`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Colorfield{B,T,M,A,H} <: AbstractField{B,T,M,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    halo_sendbuf::H
    halo_recvbuf::H
    pad::Int64 # halo width
    
    # Meta-data regarding portion of the field local to current process
    my_NX::Int64
    my_NY::Int64
    my_NZ::Int64
    my_NT::Int64
    my_NV::Int64
    nprocs::Int64
    nprocs_cart::NTuple{4,Int64}
    myrank::Int64
    myrank_cart::NTuple{4,Int64}
    comm_cart::Utils.Comm
    function Colorfield{B,T}(NX, NY, NZ, NT) where {B,T}
        U = KA.zeros(B(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        halo_sendbuf = halo_recvbuf = nothing
        pad = 0
        nprocs = 1
        nprocs_cart = (1, 1, 1, 1)
        myrank_cart = (0, 0, 0, 0)
        comm_cart = mpi_comm()
        return new{B,T,false,typeof(U),typeof(halo_sendbuf)}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, NX, NY, NZ, NT, NV,
            nprocs, nprocs_cart, mpi_myrank(), myrank_cart, comm_cart,
        )
    end

    function Colorfield{B,T}(NX, NY, NZ, NT, nprocs_cart, pad) where {B,T}
        sum(nprocs_cart) == 1 && return Colorfield{B,T}(NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        nX, nY, nZ, nT = nprocs_cart
        nprocs = nX * nY * nZ * nT
        @assert (NX%nX, NY%nY, NZ%nZ, NT%nT) == (0, 0, 0, 0) "Lattice dimensions must be divisible by comm dimensions"
        my_NX, my_NY, my_NZ, my_NT = (NX÷nX, NY÷nY, NZ÷nZ, NT÷nT)
        my_NV = my_NX * my_NY * my_NZ * my_NT
        comm_cart = mpi_cart_create(nprocs_cart; periodic=map(_->true, nprocs_cart))
        myrank_cart = (mpi_cart_coords(comm_cart, mpi_myrank())...,)
        max_halo_dims = maximum([prod(halo_dims((NX, NY, NZ, NT), pad, i)) for i in 1:4])
        halo_sendbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        halo_recvbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        U = KA.zeros(B(), SU{3,9,T}, 4, my_NX+2pad, my_NY+2pad, my_NZ+2pad, my_NT+2pad)
        return new{B,T,true,typeof(U),typeof(halo_sendbuf)}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, my_NX, my_NY, my_NZ,
            my_NT, my_NV, nprocs, nprocs_cart, mpi_myrank(), myrank_cart, comm_cart,
        )
    end
end

function Colorfield(u::AbstractField{B,T,M}) where {B,T,M}
    u_out = if M
        Colorfield{B,T}(u.NX, u.NY, u.NZ, u.NT, u.nprocs_cart, u.pad)
    else
        Colorfield{B,T}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
end

"""
    Expfield{Backend,FloatType,MPIParallel,ArrayType,HaloType} <: AbstractField{Backend,FloatType,MPIParallel,ArrayType}

5-dimensional dense array of `exp_iQ_su3` objects contatining associated meta-data. The
objects hold the `Q`-matrices and all the exponential parameters needed for stout-force
recursion.

    Expfield{Backend,FloatType}(NX, NY, NZ, NT)
    Expfield{Backend,FloatType}(NX, NY, NZ, NT, nprocs_cart, pad)
    Expfield(u::AbstractField)

Creates a Expfield on `Backend`, i.e. an array of `FloatType`-precison `exp_iQ_su3` objects
of size `4 × NX × NY × NZ × NT` or of the same size as `u`.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Expfield{B,T,M,A,H} <: AbstractField{B,T,M,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    halo_sendbuf::H
    halo_recvbuf::H
    pad::Int64 # halo width
    
    # Meta-data regarding portion of the field local to current process
    my_NX::Int64
    my_NY::Int64
    my_NZ::Int64
    my_NT::Int64
    my_NV::Int64
    nprocs::Int64
    nprocs_cart::NTuple{4,Int64}
    myrank::Int64
    myrank_cart::NTuple{4,Int64}
    comm_cart::Utils.Comm
    function Expfield{B,T}(NX, NY, NZ, NT) where {B,T}
        U = KA.zeros(B(), exp_iQ_su3{T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        halo_sendbuf = halo_recvbuf = nothing
        pad = 0
        nprocs = 1
        nprocs_cart = (1, 1, 1, 1)
        myrank_cart = (0, 0, 0, 0)
        comm_cart = mpi_comm()
        return new{B,T,false,typeof(U),typeof(halo_sendbuf)}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, NX, NY, NZ, NT, NV,
            nprocs, nprocs_cart, mpi_myrank(), myrank_cart, comm_cart,
        )
    end

    function Expfield{B,T}(NX, NY, NZ, NT, nprocs_cart, pad) where {B,T}
        sum(nprocs_cart) == 1 && return Expfield{B,T}(NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        nX, nY, nZ, nT = nprocs_cart
        nprocs = nX * nY * nZ * nT
        @assert (NX%nX, NY%nY, NZ%nZ, NT%nT) == (0, 0, 0, 0) "Lattice dimensions must be divisible by comm dimensions"
        my_NX, my_NY, my_NZ, my_NT = (NX÷nX, NY÷nY, NZ÷nZ, NT÷nT)
        my_NV = my_NX * my_NY * my_NZ * my_NT
        comm_cart = mpi_cart_create(nprocs_cart; periodic=map(_->true, nprocs_cart))
        myrank_cart = (mpi_cart_coords(comm_cart, mpi_myrank())...,)
        max_halo_dims = maximum([prod(halo_dims((NX, NY, NZ, NT), pad, i)) for i in 1:4])
        halo_sendbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        halo_recvbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        U = KA.zeros(B(), exp_iQ_su3{T}, 4, my_NX+2pad, my_NY+2pad, my_NZ+2pad, my_NT+2pad)
        return new{B,T,true,typeof(U),typeof(halo_sendbuf)}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, my_NX, my_NY, my_NZ,
            my_NT, my_NV, nprocs, nprocs_cart, mpi_myrank(), myrank_cart, comm_cart,
        )
    end
end

function Expfield(u::AbstractField{B,T,M}) where {B,T,M}
    u_out = if M
        Expfield{B,T}(u.NX, u.NY, u.NZ, u.NT, u.nprocs_cart, u.pad)
    else
        Expfield{B,T}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
end

gauge_action(::Gaugefield{B,T,M,A,GA}) where {B,T,M,A,GA} = GA

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
