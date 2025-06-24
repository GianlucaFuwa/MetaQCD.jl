"""
5-dimensional dense array of `ExpiQCoeffs` objects contatining associated meta-data. The
objects hold the `Q`-matrices and all the exponential parameters needed for stout-force
recursion.

    Expfield{B,T}(NX, NY, NZ, NT)
    Expfield{B,T}(NX, NY, NZ, NT, numprocs_cart, halo_width)
    Expfield(u::AbstractField)

Creates a Expfield on `B`, i.e. an array of `T`-precison `ExpiQCoeffs` objects
of size `4 × NX × NY × NZ × NT` or of the same size as `u`.
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Expfield{B,T,M,AT,HT,BT,TT} <: AbstractField{B,T,M,AT}
    U::AT # Actual field storing the gauge variables
    halos::HT
    sendbuf::BT
    topology::TT # Info regarding MPI topology
    function Expfield{B,T,M}(
        U::AT, halos::HT, sendbuf::BT, topology::TT
    ) where {B,T,M,AT,HT,BT,TT}
        check_types(B, T, U, halos, sendbuf)
        return new{B,T,M,AT,HT,BT,TT}(U, halos, sendbuf, topology)
    end
end

@field_constructor Expfield

function Expfield(
    u::AbstractField{B,T,M}; no_halo=false, hw=get_halo_width(u)
) where {B,T,M}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Expfield{B,T}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Expfield{B,T}(size(u)...)
    end

    return u_out
end

Base.eltype(::Type{Expfield}, ::Type{T}) where {T} = ExpiQCoeffs{T}
