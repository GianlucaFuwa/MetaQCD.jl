struct Gaugefield{Backend,IsMPI,FloatT,A,H,GaugeAction} <: Abstractfield{Backend,IsMPI,FloatT,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    halo_send::H
    halo_recv::H
    
    # Meta-data regarding portion of the field local to current process
    my_NX::Int64
    my_NY::Int64
    my_NZ::Int64
    my_NT::Int64
    my_NV::Int64
    nprocs::Int64
    nprocs_cart::NTuple{4,Int64}
    myrank::Int64
    myrank_coords::NTuple{4,Int64}
    comm_cart::MPI.Comm

    β::Float64 # Seems weird to have it here, but I couldnt be bothered passing it as an argument everywhere
    Sg::Base.RefValue{Float64} # Current Gauge action, used to safe work
    CV::Base.RefValue{Float64} # Current collective variable, used to safe work
    function Gaugefield{BACKEND,T,GA}(NX, NY, NZ, NT, β) where {BACKEND,T,GA}
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        halo_send = halo_recv = nothing
        nprocs = 1
        nprocs_cart = (1, 1, 1, 1)
        myrank_coords = (0, 0, 0, 0)
        comm_cart = MPI.Cart_create(COMM, nprocs_cart; periodic=map(_->true, dims))
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{BACKEND,T,false,typeof(U),GA}(
            U, NX, NY, NZ, NT, NV, 3, halo_send, halo_recv, NX, NY, NZ, NT, NV, nprocs,
            nprocs_cart, MYRANK, myrank_coords, comm_cart, β, Sg, CV,
        )
    end

    function Gaugefield{BACKEND,T,GA}(NX, NY, NZ, NT, nprocs_cart, β) where {BACKEND,T,GA}
        prod(nprocs_cart) == 1 && return Gaugefield{BACKEND,T,GA}(NX, NY, NZ, NT, β)
        NV = NX * NY * NZ * NT
        nX, nY, nZ, nT = nprocs_cart
        nprocs = prod((nX, nY, nZ, nT))
        @assert nprocs == COMM_SIZE
        @assert (NX%nX, NY%nY, NZ%nZ, NT%nT) == (0, 0, 0, 0) "Lattice dimensions must be divisible by comm dimensions"
        my_NX, my_NY, my_NZ, my_NT = (NX÷nX, NY÷nY, NZ÷nZ, NT÷nT)
        comm_cart = MPI.Cart_create(COMM, (1, 1, 1, 1); periodic=map(_->true, dims))
        comm_cart = MPI.Cart_create(COMM, (nX, nY, nZ, nT); periodic=map(_->true, dims))
        pad = 1
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, my_NX+2pad, my_NY+2pad, my_NZ+2pad, my_NT+2pad)
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{BACKEND,T,true,typeof(U),GA}(
            U, NX, NY, NZ, NT, NV, 3, halo_send, halo_recv, NX, NY, NZ, NT, NV, nprocs,
            nprocs_cart, MYRANK, myrank_coords, comm_cart, β, Sg, CV,
        )
    end
end

function Gaugefield(parameters)
    NX, NY, NZ, NT = parameters.L
    β = parameters.beta
    GA = GAUGE_ACTION[parameters.gauge_action]
    T = Utils.FLOAT_TYPE[parameters.float_type]
    B = BACKENDS[parameters.backend]
    nprocs_cart = parameters.nprocs_cart
    nprocs = prod(nprocs_cart)

    U = if nprocs > 1
        Gaugefield{B,T,GA}(NX, NY, NZ, NT, nprocs_cart, β)
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
