@field_constructor Expfield

"""
Wrapper around a 5-dimensional dense array of `ExpiQCoeffs` objects contatining
information about the global MPI-topology. The array holds the `Q`-matrices and all the
exponential parameters needed for stout-force recursion.

    Expfield{B,T}(NX, NY, NZ, NT)
    Expfield{B,T}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Expfield(u::AbstractField)

Creates a Expfield on `B`, i.e. an array of `T`-precison `ExpiQCoeffs` objects
of size `4 × NX × NY × NZ × NT` or of the same size as `u`.
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Expfield

function Expfield(
    u::AbstractField{B,T,M}, ::Type{Tnew}=T; no_halo=false, hw=get_halo_width(u)
) where {B,T,M,Tnew}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Expfield{B,Tnew}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Expfield{B,Tnew}(size(u)...)
    end

    return u_out
end

Base.eltype(::Type{Expfield}, ::Type{T}) where {T} = ExpiQCoeffs{T}
