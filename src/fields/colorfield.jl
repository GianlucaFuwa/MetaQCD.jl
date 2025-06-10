"""
5-dimensional dense array of statically sized 3x3 matrices contatining associated meta-data.

    Colorfield{Backend,FloatType}(NX, NY, NZ, NT)
    Colorfield{Backend,FloatType}(NX, NY, NZ, NT, numprocs_cart, halo_width; nohalo=false)
    Colorfield(u::AbstractField)

Creates a Colorfield on `Backend`, i.e. an array of 3-by-3 `FloatType`-precision matrices of
size `4 × NX × NY × NZ × NT` or a zero-initialized Colorfield of the same size as `u`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Colorfield{Backend,FloatType,IsDistributed,ArrayType,BufferType} <:
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
    function Colorfield{B,T,M,AT,BT}(
        U::AT, send_buf::BT, recv_buf::BT, NX, NY, NZ, NT, NV, NC, topology
    ) where {B,T,M,AT,BT}
        # some sanity checks
        @assert get_backend(U) isa Backend
        if BT !== Nothing
            @assert eltype(U) == eltype(send_buf) == eltype(recv_buf)
        end
        @assert eltype(eltype(U)) === Complex{T}
        return new{B,T,M,AT,BT}(
            U, send_buf, recv_buf, NX, NY, NZ, NT, NV, NC, topology
        )
    end
end

function Colorfield{Backend,FloatType}(NX, NY, NZ, NT) where {Backend,FloatType}
    U = KA.zeros(Backend(), SU{3,9,FloatType}, 4, NX, NY, NZ, NT)
    send_buf = recv_buf = nothing
    NV = NX * NY * NZ * NT
    numprocs_cart = (1, 1, 1, 1)
    halo_width = 0
    topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
    return Colorfield{Backend,FloatType,false,typeof(U),Nothing}(
        U, send_buf, recv_buf, NX, NY, NZ, NT, NV, 3, topology
    )
end

function Colorfield{Backend,FloatType}(
    NX, NY, NZ, NT, numprocs_cart, halo_width; nohalo=false
) where {Backend,FloatType}
    if prod(numprocs_cart) == 1
        return Colorfield{Backend,FloatType}(NX, NY, NZ, NT)
    end

    NV = NX * NY * NZ * NT
    topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
    ldims = nohalo ? topology.local_dims : topology.local_dims_padded

    eff_halo_width = halo_width .* topology.is_partitioned
    origin = OffsetArrays.Origin((1, topology.bulk_sites[1].I .- eff_halo_width...)...)
    U = OffsetArray(KA.zeros(Backend(), SU{3,9,FloatType}, 4, ldims...), origin)
    buf_length = 4maximum(sz for sz in topology.halo_sizes)
    send_buf = nohalo ? nothing : KA.zeros(Backend(), SU{3,9,FloatType}, buf_length)
    recv_buf = nohalo ? nothing : KA.zeros(Backend(), SU{3,9,FloatType}, buf_length)
    return Colorfield{Backend,FloatType,true,typeof(U),typeof(send_buf)}(
        U, send_buf, recv_buf, NX, NY, NZ, NT, NV, 3, topology
    )
end

function Colorfield(
    u::AbstractField{Backend,FloatType,IsDistributed}; nohalo=false
) where {Backend,FloatType,IsDistributed}
    u_out = if IsDistributed
        numprocs_cart = u.topology.numprocs_cart
        halo_width = maximum(u.topology.halo_width)
        Colorfield{Backend,FloatType}(
            u.NX, u.NY, u.NZ, u.NT, numprocs_cart, halo_width; nohalo=nohalo
        )
    else
        Colorfield{Backend,FloatType}(u.NX, u.NY, u.NZ, u.NT)
    end

    return u_out
end

