# INFO: This struct is here in case I ever want to use a more efficient storage format for
# algebra elements
struct Algebrafield{Backend,FloatType,IsDistributed,ArrayType} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    
    topology::FieldTopology # Info regarding MPI topology
    function Algebrafield{Backend,FloatType}(NX, NY, NZ, NT) where {Backend,FloatType}
        U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        numprocs_cart = (1, 1, 1, 1)
        halo_width = 0
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        return new{Backend,FloatType,false,typeof(U)}(U, NX, NY, NZ, NT, NV, 3, topology)
    end

    function Algebrafield{Backend,FloatType}(NX, NY, NZ, NT, numprocs_cart, halo_width) where {Backend,FloatType}
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

function Algebrafield(
    u::AbstractField{Backend,FloatType,IsDistributed}
) where {Backend,FloatType,IsDistributed}
    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = u.topology.halo_width
        Algebrafield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT, numprocs_cart, halo_width)
    else
        Algebrafield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT)
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
