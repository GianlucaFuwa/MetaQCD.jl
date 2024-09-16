abstract type AbstractFieldstrength end

struct Plaquette <: AbstractFieldstrength end
struct Clover <: AbstractFieldstrength end
struct Improved <: AbstractFieldstrength end

# TODO: Docs
struct Tensorfield{Backend,FloatType,IsDistributed,ArrayType} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    U::ArrayType # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    
    topology::FieldTopology # Info regarding MPI topology
    function Tensorfield{Backend,FloatType}(NX, NY, NZ, NT) where {Backend,FloatType}
        U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        numprocs_cart = (1, 1, 1, 1)
        halo_width = 0
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        return new{Backend,FloatType,false,typeof(U)}(U, NX, NY, NZ, NT, NV, 3, topology)
    end

    function Tensorfield{Backend,FloatType}(
        NX, NY, NZ, NT, numprocs_cart, halo_width
    ) where {Backend,FloatType}
        if prod(numprocs_cart) == 1
            return Tensorfield{Backend,FloatType}(NX, NY, NZ, NT)
        end

        NV = NX * NY * NZ * NT
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
        ldims = topology.local_dims
        dims_in = ntuple(i -> ldims[i]+2halo_width, Val(4)) 
        U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, 4, dims_in...)
        return new{Backend,FloatType,true,typeof(U)}(U, NX, NY, NZ, NT, NV, 3, topology)
    end
end

function Tensorfield(
    u::Gaugefield{Backend,FloatType,IsDistributed,ArrayType}
) where {Backend,FloatType,IsDistributed,ArrayType}
    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = u.topology.halo_width
        Tensorfield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT, numprocs_cart, halo_width)
    else
        Tensorfield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
end

# overload get and set for the Tensorfields, so we dont have to do u.U[μ,ν,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Tensorfield, μ, ν, x, y, z, t) =
    u.U[μ, ν, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::Tensorfield, μ, ν, site::SiteCoords) =
    u.U[μ, ν, site]
Base.@propagate_inbounds Base.setindex!(u::Tensorfield, v, μ, ν, x, y, z, t) =
    setindex!(u.U, v, μ, ν, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::Tensorfield, v, μ, ν, site::SiteCoords) =
    setindex!(u.U, v, μ, ν, site)

Base.view(u::Tensorfield, I::CartesianIndices{4}) = view(u.U, 1:4, 1:4, I.indices...)

function fieldstrength_eachsite!(F::Tensorfield, U, kind_of_fs::String)
    if kind_of_fs == "plaquette"
        fieldstrength_eachsite!(Plaquette(), F, U)
    elseif kind_of_fs == "clover"
        fieldstrength_eachsite!(Clover(), F, U)
    else
        error("kind of fieldstrength \"$(kind_of_fs)\" not supported")
    end

    return nothing
end

function fieldstrength_eachsite!(::Plaquette, F::Tensorfield{CPU}, U::Gaugefield{CPU})
    check_dims(F, U)

    @batch for site in eachindex(U)
        C12 = plaquette(U, 1, 2, site)
        F[1, 2, site] = im * traceless_antihermitian(C12)
        C13 = plaquette(U, 1, 3, site)
        F[1, 3, site] = im * traceless_antihermitian(C13)
        C14 = plaquette(U, 1, 4, site)
        F[1, 4, site] = im * traceless_antihermitian(C14)
        C23 = plaquette(U, 2, 3, site)
        F[2, 3, site] = im * traceless_antihermitian(C23)
        C24 = plaquette(U, 2, 4, site)
        F[2, 4, site] = im * traceless_antihermitian(C24)
        C34 = plaquette(U, 3, 4, site)
        F[3, 4, site] = im * traceless_antihermitian(C34)
    end

    update_halo!(F)
    return nothing
end

function fieldstrength_eachsite!(
    ::Clover, F::Tensorfield{CPU,T}, U::Gaugefield{CPU,T}
) where {T}
    check_dims(F, U)
    fac = Complex{T}(im / 4)

    @batch for site in eachindex(U)
        C12 = clover_square(U, 1, 2, site, 1)
        F[1, 2, site] = fac * traceless_antihermitian(C12)
        C13 = clover_square(U, 1, 3, site, 1)
        F[1, 3, site] = fac * traceless_antihermitian(C13)
        C14 = clover_square(U, 1, 4, site, 1)
        F[1, 4, site] = fac * traceless_antihermitian(C14)
        C23 = clover_square(U, 2, 3, site, 1)
        F[2, 3, site] = fac * traceless_antihermitian(C23)
        C24 = clover_square(U, 2, 4, site, 1)
        F[2, 4, site] = fac * traceless_antihermitian(C24)
        C34 = clover_square(U, 3, 4, site, 1)
        F[3, 4, site] = fac * traceless_antihermitian(C34)
    end

    update_halo!(F)
    return nothing
end

function fieldstrength_A_eachsite!(
    ::Clover, F::Tensorfield{CPU,T}, U::Gaugefield{CPU,T}
) where {T}
    check_dims(F, U)
    fac = Complex{T}(im / 4)

    @batch for site in eachindex(U)
        C12 = clover_square(U, 1, 2, site, 1)
        F[1, 2, site] = fac * antihermitian(C12)
        C13 = clover_square(U, 1, 3, site, 1)
        F[1, 3, site] = fac * antihermitian(C13)
        C14 = clover_square(U, 1, 4, site, 1)
        F[1, 4, site] = fac * antihermitian(C14)
        C23 = clover_square(U, 2, 3, site, 1)
        F[2, 3, site] = fac * antihermitian(C23)
        C24 = clover_square(U, 2, 4, site, 1)
        F[2, 4, site] = fac * antihermitian(C24)
        C34 = clover_square(U, 3, 4, site, 1)
        F[3, 4, site] = fac * antihermitian(C34)
    end

    update_halo!(F)
    return nothing
end
