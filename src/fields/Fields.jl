module Fields

using Base.Meta: quot
using LinearAlgebra
using MacroTools
using OffsetArrays
using Polyester # Used for the @batch macro, which enables multi threading
using Preferences
using Random
using SIMD
using StaticArrays # Used for the SU3 matrices
using ..Utils # Contains utility functions, such as projections and the exponential map

import Adapt: adapt_structure
import StrideArraysCore: PtrArray, object_and_preserve # This is used to convert the AbstractField to a PtrArray in the @batch loop

struct CPU end
# When CUDA.jl or AMDGPU.jl are loaded, their backends are appended to this Dict
const BACKENDS = Dict{String,Any}("cpu" => CPU)
const DEVICE_ID = Base.RefValue{Int64}(-1)
const HOSTNAME = Val(Symbol(gethostname()))
const FORCE_SINGLE_GPU = Val(@load_preference("FORCE_SINGLE_GPU", false))

# We are going to need these if we want to transfer a field from one backend to another
# For other backends, we overload this method in their respective extensions
array_type(::Type{CPU}) = Array
bzeros(::CPU, args...) = zeros(args...)
synchronize(::CPU) = nothing
synchronize(::CPU, stream) = nothing
device_synchronize(::CPU) = nothing
priority!(::CPU, priority) = nothing
mpi_assign_device!(::CPU, id) = nothing
allocate_commstreams!(::CPU, args...) = nothing
default_stream(::CPU) = nothing
get_readstream(::CPU, args...) = nothing
get_sendstream(::CPU, args...) = nothing
get_stream(::CPU, args...) = nothing
get_priority_stream(::CPU, args...) = nothing

# Define an abstract field super type that is parametrized by the backend, the precision and
# the array type (Array, CuArray, ROCArray)
abstract type AbstractField{Backend,FloatType,IsDistributed} end

const AbstractMPIField{Backend,FloatType} = AbstractField{Backend,FloatType,true}

@inline function is_distributed(
    ::AbstractField{Backend,FloatType,IsDistributed}
) where {Backend,FloatType,IsDistributed}
    return IsDistributed
end

@inline is_evenodd(::AbstractField) = false # is only true for SpinorfieldEO

# utility functions for MPI-distributed fields
include("distributed/topology.jl")
include("distributed/halo_update_async.jl")
# include("distributed/halo_update_async_gpu.jl")
# include("distributed/hide_communication.jl")
include("distributed/comm_utils.jl")

include("utils/parallel.jl")
include("utils/layout_cpu.jl")
include("utils/layout_gpu.jl")
include("utils/constructor.jl")
include("utils/boundaries.jl") # boundary conditions in time direction for spinors
include("gaugefield.jl") # Gaugefield, Colorfield and Expfield structs defined here
include("colorfield.jl")
include("expfield.jl")
include("algebrafield.jl") # For now just a placeholder in case I want to implement more efficient storage of su(3) algebra elements
include("spinorfield.jl") # Spinorfield structs defined here 
include("spinorfield_eo.jl") # Spinorfield for even-odd precon
include("multispinorfield.jl") # MultiSpinorfield structs defined here 
include("paulifield.jl") # For now just a placeholder in case I want to implement more efficient storage of su(3) algebra elements
include("tensorfield.jl") # Tensorfield struct and fieldstrength methods defined here
include("utils/iterators.jl") # Sequential and Checkerboard iterators defined here 
include("utils/adapt.jl")

const GaugeLikeField{B,T,M} = Union{Gaugefield{B,T,M},Colorfield{B,T,M}}

# XXX: Not sure why these are here, but whatever
Base.similar(u::Gaugefield{B,T}, ::Type{Tnew}=T) where {B,T,Tnew} = Gaugefield(u, Tnew)
Base.similar(u::Colorfield{B,T}, ::Type{Tnew}=T) where {B,T,Tnew} = Colorfield(u, Tnew)
Base.similar(u::Expfield{B,T}, ::Type{Tnew}=T) where {B,T,Tnew} = Expfield(u, Tnew)
Base.similar(u::Tensorfield{B,T}, ::Type{Tnew}=T) where {B,T,Tnew} = Tensorfield(u, Tnew)
Base.similar(u::Spinorfield{B,T}, ::Type{Tnew}=T) where {B,T,Tnew} = Spinorfield(u, Tnew)
Base.similar(u::MultiSpinorfield{B,T}, ::Type{Tnew}=T) where {B,T,Tnew} = MultiSpinorfield(u, Tnew)
Base.similar(u::SpinorfieldEO) = SpinorfieldEO(u.parent)

"""
    convert_field(Backend_out, u::AbstractField{CPU,FloatType_in}, ::Type{FloatType_out})

Ports the AbstractField u from CPU to `Backend_out`, maintaining all elements
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
function convert_field(
    ::Type{Bout}, uin::AbstractField{CPU,Tin,M}, ::Type{Tout}=Tin
) where {M,Bout,Tout,Tin}
    # This version is only for Colorfield and Tensorfield
    # the others are defined in their respective files
    if Bout === CPU
        uout = similar(uin, Tout)
        copy!(uout, uin)
        return uout
    end

    Fieldtype = eval(nameof(typeof(uin)))
    NX, NY, NZ, NT = size(uin)
    numprocs_cart = get_numprocs_cart(uin)
    halo_width = get_halo_width(uin)
    uout = Fieldtype{Bout,Tout}(NX, NY, NZ, NT; numprocs_cart, halo_width)
    uarr = array_type(Bout)(uin.U)
    inner_length = if uin isa Colorfield
        4
    elseif uin isa Tensorfield
        6
    end

    parallelfor(eachindex(uout), Bout, Val(M), (uout,), (), (uout,)) do site, (uout,)
        for i in 1:inner_length
            uout[i, site] = uarr[i, site]
        end
    end

    return uout
end

function device_to_host(x, ::Type{B}) where {B}
    if x isa array_type(B)
        return x
    else
        return array_type(B)(x)
    end
end

# XXX: might not be needed
Base.eltype(u::AbstractField) = eltype(u.U)
Base.elsize(u::AbstractField) = Base.elsize(u.U)
Base.parent(u::AbstractField) = u.U
Base.pointer(u::AbstractField) = pointer(u.U)
Base.strides(u::AbstractField) = strides(u.U)

# Some useful functions that share geometry information about the fields
get_backend(::AbstractField{B}) where B = B
@inline Base.length(u::AbstractField) = u.topology.global_volume
@inline Base.size(u::AbstractField) = u.topology.global_dims
@inline Base.size(u::AbstractField, μ) = u.topology.global_dims[μ]
@inline Base.axes(u::AbstractField) = u.topology.bulk_sites_padded.indices
@inline Base.axes(u::AbstractField, μ::Integer) = u.topology.bulk_sites_padded.indices[μ]
# Base.axes(u::AbstractField, μ::Integer) = u.topology.bulk_sites.indices[μ]
@inline float_type(::AbstractField{B,T}) where {B,T} = T
@inline num_colors(u::AbstractField) = 3
@inline get_local_dims(u::AbstractField) = u.topology.local_dims
@inline get_local_volume(u::AbstractField) = u.topology.local_volume
@inline get_global_volume(u::AbstractField) = u.topology.global_volume
@inline get_halo_width(u::AbstractField) = u.topology.halo_width
@inline get_numprocs_cart(u::AbstractField) = u.topology.numprocs_cart

@inline function invalidate_halo!(u::AbstractField) 
    if u.halo_valid isa Base.RefValue{Bool}
        u.halo_valid[] = false
    end

    return nothing
end

@inline function validate_halo!(u::AbstractField)
    if u.halo_valid isa Base.RefValue{Bool}
        u.halo_valid[] = false
    end

    return nothing
end

@inline function halo_is_valid(u::AbstractField)
    return u.halo_valid isa Base.RefValue{Bool} ? u.halo_valid[] : true
end

# Field iterators / lattice sites
@inline function Base.eachindex(u::AbstractField, fields...)
    check_dims(u, fields...)
    return eachindex(u)
end

@inline function Base.eachindex(arg::Union{Symbol,Bool}, u::AbstractField, fields...)
    check_dims(u, fields...)
    return eachindex(arg, u)
end

@inline Base.eachindex(u::AbstractField) = u.topology.bulk_sites
@inline Base.eachindex(::IndexLinear, u::AbstractField) = Base.OneTo(u.topology.local_volume)

@inline function Base.eachindex(parity::Symbol, u::AbstractField)
    return u.topology.bulk_sites_eo[parity]
end

@inline function Base.eachindex(even_half::Bool, u::AbstractField)
    bulk_s..., bulk_t = u.topology.bulk_sites.indices
    nt = length(bulk_t)
    t_start = first(bulk_t)
    t_stop = last(bulk_t)
    @assert iseven(nt) "length in time dimension needs to be even for even-odd precon"
    nt2 = nt ÷ 2
    last_range = even_half ? (t_start:t_start+nt2-1) : (t_start+nt2:t_stop)
    return CartesianIndices((bulk_s..., last_range))
end

@inline function allindices(u::AbstractField, fields...)
    @assert all(x -> allindices(x) == allindices(u), fields)
    return allindices(u)
end

@inline allindices(u::AbstractField{B}) where {B} = add_directional_indices(u, eachindex(u))

@inline function linear_index(u::AbstractField, idx::CartesianIndex{N}) where N
    U = u.U
    axs = axes(U)
    sz = size(U)
    lin_idx = 1
    stride = 1

    for i in 1:N
        # Normalize index to 1-based
        norm_idx = idx[i] - first(axs[i]) + 1
        lin_idx += (norm_idx - 1) * stride
        stride *= sz[i]
    end

    return lin_idx
end

Base.@propagate_inbounds Base.getindex(u::AbstractField, i::Integer) = u.U[i]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μsite) = u.U[μsite]
Base.@propagate_inbounds Base.getindex(u::AbstractField{CPU}, μ, site) = u.U[μ, site]
Base.@propagate_inbounds Base.getindex(u::AbstractField{B}, μ, site) where {B} = u.U[site, μ]

Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, i::Integer) =
    setindex!(u.U, v, i)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μsite) =
    setindex!(u.U, v, μsite)
Base.@propagate_inbounds Base.setindex!(u::AbstractField{CPU}, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)
Base.@propagate_inbounds Base.setindex!(u::AbstractField{B}, v, μ, site::SiteCoords) where {B} =
    setindex!(u.U, v, site, μ)

function get_recv_buf(u::AbstractField{B,T,M}, num) where {B,T,M}
    @assert 1 <= num <= 8 "recvbuf index $num is out-of-bounds (must be in [1, 8])"
    return mpi_make_transferrable(u.recvbuf[num])
end

function check_types(::Type{B}, ::Type{T}, U, sendbuf, recvbuf) where {B,T}
    # some sanity checks
    # if !(U isa PtrArray)
    #     @assert get_backend(U) isa B
    # end
    #
    # if !isnothing(sendbuf)
    #     if !(sendbuf isa PtrArray)
    #         @assert get_backend(sendbuf) isa B
    #     end
    # end
    # if !isnothing(recvbuf)
    #     if !(sendbuf isa PtrArray)
    #         @assert get_backend(recvbuf) isa B
    #     end
    # end
    # @assert eltype(eltype(U)) === Complex{T}
    return nothing
end

"""
    check_dims(x1, rest...)

Check if all fields have the same dimensions. Throw an `AssertionError` otherwise.
"""
@generated function check_dims(x1, rest::Vararg{Any,N}) where {N}
    q_inner = Expr(:comparison, :(size(x1)))

    for i in 1:N
        push!(q_inner.args, :(==))
        push!(q_inner.args, :(size(rest[$i])))
    end

    q = Expr(:macrocall, Symbol("@assert"), :(), q_inner)
    return q
end

include("field_operations.jl") # General operations on fields, like adding, copying etc.
include("action.jl") # Gauge action methods
include("stencils/plaquette.jl") # Definition of clover operator
include("stencils/clover.jl") # Definition of clover operator
include("stencils/staple.jl") # Definition of staple operator
include("stencils/wilsonloop.jl") # Definition of arbitrary side length Wilson loops

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", u::AbstractField{B,T}) where {B,T}
    print(io, "$(nameof(typeof(u))){$B,$T}", "(\n")
    println(io, "\tsize:", " $(size(u))")

    if u isa Spinorfield || u isa MultiSpinorfield
        println(io, "\tnumdirac:", " $(num_dirac(u))")
    end

    u isa MultiSpinorfield && println(io, "\tnumspinors:", " $(num_spinors(u))")

    for fieldname in fieldnames(typeof(u))
        if fieldname in (:U, :sendbuf, :recvbuf)
            println(io, "\t", fieldname, ": $(nameof(typeof(getfield(u, fieldname))))")
        elseif fieldname == :topology
            println(io, "\t", fieldname, ": FieldTopology")
        else
            println(io, "\t", fieldname, ": ", getfield(u, fieldname))
        end
    end

    print(io, ")")
    return nothing
end

function Base.show(io::IO, u::AbstractField{B,T}) where {B,T}
    print(io, "$(nameof(typeof(u))){$B,$T}", "(\n")
    println(io, "\tsize:", " $(size(u))")

    if u isa Spinorfield || u isa MultiSpinorfield
        println(io, "\tnumdirac:", " $(num_dirac(u))")
    end

    u isa MultiSpinorfield && println(io, "\tnumspinors:", " $(num_spinors(u))")

    for fieldname in fieldnames(typeof(u))
        if fieldname in (:U, :sendbuf, :recvbuf)
            println(io, "\t", fieldname, ": $(nameof(typeof(getfield(u, fieldname))))")
        elseif fieldname == :topology
            println(io, "\t", fieldname, ": FieldTopology")
        else
            println(io, "\t", fieldname, ": ", getfield(u, fieldname))
        end
    end

    print(io, ")")
    return nothing
end

end
