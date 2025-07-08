@field_constructor Colorfield

@doc raw"""
Wrapper around a 5-dimensional dense array of statically sized 3x3 matrices contatining
information about the global MPI-topology.

    Colorfield{B,T}(NX, NY, NZ, NT)
    Colorfield{B,T}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    Colorfield(u::AbstractField)

Creates a Colorfield on `B`, i.e. an array of 3-by-3 `T`-precision matrices of
size `4 × NX × NY × NZ × NT` or a zero-initialized Colorfield of the same size as `u`
# Supported backends
`CPU` \
`CUDABackend` (provided CUDA.jl is loaded) \
`ROCBackend` (provided AMDGPU.jl is loaded)
""" Colorfield

function Colorfield(
    u::AbstractField{B,T,M}; no_halo=false, hw=get_halo_width(u)
) where {B,T,M}
    u_out = if M
        ncart = get_numprocs_cart(u)
        Colorfield{B,T}(size(u)...; numprocs_cart=ncart, halo_width=hw, no_halo=no_halo)
    else
        Colorfield{B,T}(size(u)...)
    end

    return u_out
end

Base.eltype(::Type{Colorfield}, ::Type{T}) where {T} = SMatrix{3,3,Complex{T},9}
