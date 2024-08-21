# INFO: This struct is here in case I ever want to use a more efficient storage format for
# algebra elements
struct Algebrafield{B,T,M,A,H} <: AbstractField{B,T,M,A}
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
    comm_cart::MPI.Comm
    function Algebrafield{B,T}(NX, NY, NZ, NT) where {B,T}
        U = KA.zeros(B(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        halo_sendbuf = halo_recvbuf = nothing
        pad = 0
        nprocs = 1
        nprocs_cart = (1, 1, 1, 1)
        myrank_cart = (0, 0, 0, 0)
        comm_cart = MPI.Cart_create(COMM, nprocs_cart; periodic=map(_->true, dims))
        return new{B,T,false,typeof(U),typeof(halo_sendbuf)}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, NX, NY, NZ, NT, NV,
            nprocs, nprocs_cart, MYRANK, myrank_cart, comm_cart,
        )
    end

    function Algebrafield{B,T}(NX, NY, NZ, NT, nprocs_cart, pad) where {B,T}
        sum(nprocs_cart) == 1 && return Algebrafield{B,T}(NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        nX, nY, nZ, nT = nprocs_cart
        nprocs = nX * nY * nZ * nT
        @assert nprocs == COMM_SIZE
        @assert (NX%nX, NY%nY, NZ%nZ, NT%nT) == (0, 0, 0, 0) "Lattice dimensions must be divisible by comm dimensions"
        my_NX, my_NY, my_NZ, my_NT = (NX÷nX, NY÷nY, NZ÷nZ, NT÷nT)
        my_NV = my_NX * my_NY * my_NZ * my_NT
        comm_cart = MPI.Cart_create(COMM, nprocs_cart; periodic=map(_->true, nprocs_cart))
        myrank_cart = (MPI.Cart_coords(comm_cart, MYRANK)...,)
        max_halo_dims = maximum([prod(halo_dims((NX, NY, NZ, NT), pad, i)) for i in 1:4])
        halo_sendbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        halo_recvbuf = KA.zeros(B(), SU{3,9,T}, 4, max_halo_dims)
        U = KA.zeros(B(), SU{3,9,T}, 4, my_NX+2pad, my_NY+2pad, my_NZ+2pad, my_NT+2pad)
        return new{B,T,true,typeof(U),typeof(halo_sendbuf)}(
            U, NX, NY, NZ, NT, NV, 3, halo_sendbuf, halo_recvbuf, pad, my_NX, my_NY, my_NZ,
            my_NT, my_NV, nprocs, nprocs_cart, MYRANK, myrank_cart, comm_cart,
        )
    end
end

function Algebrafield(u::AbstractField{B,T,M}) where {B,T,M}
    u_out = if M
        Algebrafield{B,T}(u.NX, u.NY, u.NZ, u.NT, u.nprocs_cart, u.pad)
    else
        Algebrafield{B,T}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
end

function gaussian_TA!(p::Algebrafield{CPU,T}, ϕ=0) where {T}
    # friction is a number in the range [0,1] instead of an angle; it's easier to use
    # have to make sure that ϕ₁² + ϕ₂² = 1
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)

    for site in eachindex(p)
        for μ in 1:4
            p[μ, site] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * p[μ, site]
        end
    end

    update_halo!(p)
    return nothing
end

function gaussian_TA!(p::Colorfield{CPU,T}, ϕ=0) where {T}
    # friction is a number in the range [0,1] instead of an angle; it's easier to use
    # have to make sure that ϕ₁² + ϕ₂² = 1
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)

    for site in eachindex(p)
        for μ in 1:4
            p[μ, site] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * p[μ, site]
        end
    end

    update_halo!(p)
    return nothing
end

function calc_kinetic_energy(p::Algebrafield{CPU})
    K = 0.0

    @batch reduction = (+, K) for site in eachindex(p)
        for μ in 1:4
            pmat = materialize_TA(p[μ, site]...)
            K += real(multr(pmat, pmat))
        end
    end

    return distributed_reduce(K, +, p)
end

function calc_kinetic_energy(p::Colorfield{CPU})
    K = 0.0

    @batch reduction = (+, K) for site in eachindex(p)
        for μ in 1:4
            pmat = p[μ, site]
            K += real(multr(pmat, pmat))
        end
    end

    return distributed_reduce(K, +, p)
end
