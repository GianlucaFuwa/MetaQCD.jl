module Fields

using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using MacroTools
using OffsetArrays
using Polyester # Used for the @batch macro, which enables multi threading
using Random
using StaticArrays # Used for the SU3 matrices
using ..Utils # Contains utility functions, such as projections and the exponential map

import KernelAbstractions as KA # With this we can write generic GPU kernels for ROC and CUDA
import StrideArraysCore: object_and_preserve # This is used to convert the AbstractField to a PtrArray in the @batch loop

# When CUDA.jl or AMDGPU.jl are loaded, their backends are appended to this Dict
const BACKENDS = Dict{String,Type{<:KA.Backend}}("cpu" => CPU)

# We are going to need these if we want to transfer a field from one backend to another
# For other backends, we overload this method in their respective extensions
@inline array_type(::Type{CPU}) = Array

# Define an abstract field super type that is parametrized by the backend, the precision and
# the array type (Array, CuArray, ROCArray)
abstract type AbstractField{Backend,FloatType,IsDistributed,ArrayType} end

const AbstractMPIField{Backend,FloatType,ArrayType} =
    AbstractField{Backend,FloatType,true,ArrayType}

@inline function is_distributed(
    ::AbstractField{Backend,FloatType,IsDistributed,ArrayType}
) where {Backend,FloatType,IsDistributed,ArrayType}
    return IsDistributed
end

@inline is_evenodd(::AbstractField) = false # is only true for SpinorfieldEO

# utility functions for MPI-distributed fields
include("distributed/topology.jl")
include("distributed/halo_update.jl")

include("boundaries.jl") # boundary conditions in time direction for spinors
include("gaugefield.jl") # Gaugefield, Colorfield and Expfield structs defined here
include("colorfield.jl")
include("expfield.jl")
include("algebrafield.jl") # For now just a placeholder in case I want to implement more efficient storage of su(3) algebra elements
include("spinorfield.jl") # Spinorfield structs defined here 
include("spinorfield_eo.jl") # Spinorfield for even-odd precon
include("multispinorfield.jl") # MultiSpinorfield structs defined here 
include("paulifield.jl") # For now just a placeholder in case I want to implement more efficient storage of su(3) algebra elements
include("tensorfield.jl") # Tensorfield struct and fieldstrength methods defined here
include("iterators/cpu_iterators.jl") # Sequential and Checkerboard iterators defined here 
include("iterators/gpu_iterators.jl") # GPU version of the above
include("gpu_kernels/utils.jl")

include("action.jl") # Gauge action methods
include("field_operations.jl") # General operations on fields, like adding, copying etc.
include("stencils/clover.jl") # Definition of clover operator
include("stencils/staple.jl") # Definition of staple operator
include("stencils/wilsonloop.jl") # Definition of arbitrary side length Wilson loops

include("gpu_kernels/action.jl") # GPU versions of the above:
include("gpu_kernels/algebrafield.jl")
include("gpu_kernels/field_operations.jl")
# TODO: include("gpu_kernels/paulifield.jl")
include("gpu_kernels/spinorfield.jl")
include("gpu_kernels/tensorfield.jl")
include("gpu_kernels/wilsonloop.jl")

Base.similar(u::Gaugefield) = Gaugefield(u)
Base.similar(u::Colorfield) = Colorfield(u)
Base.similar(u::Expfield) = Expfield(u)

Base.view(u::AbstractField, I::CartesianIndices{4}) = view(u.U, 1:4, I.indices...)
Base.view(u::AbstractField, I::Vector{CartesianIndex{4}}) = view(u.U, 1:4, I)

"""
    to_backend(Backend_out, u::AbstractField{Backend_in,FloatType})

Ports the AbstractField u to the backend `Backend_out`, maintaining all elements
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
function to_backend(
    ::Type{Bout}, u::AbstractField{Bin,Tin,M}, ::Type{Tout}=Tin
) where {M,Bout,Tout,Bin,Tin}
    @assert M === false "Switching backends not yet supported with MPI parallelization" # FIXME

    if Bout === Bin
        u_out = similar(u)
        copy!(u_out, u)
        return u_out
    end

    A = array_type(Bout)
    new_eltype = convert(Tout, eltype(u.U))
    sizeU = global_dims(u)
    Uout = A{new_eltype}(u.U)
    bufs = if isnothing(u.send_buf)
        nothing, nothing
    else
        A{new_eltype}(u.send_buf), A(u.recv_buf)
    end
    Fieldtype = eval(nameof(typeof(u)))

    if u isa Gaugefield
        GA = gauge_action(u)
        Sg = Base.RefValue{Float64}(u.Sg)
        CV = deepcopy(u.CV)
        return Gaugefield{Bout,Tout,M,typeof(Uout),GA,typeof(bufs[1])}(
            Uout, bufs..., sizeU..., u.NV, 3, u.topology, u.β, Sg, CV
        )
    elseif u isa Spinorfield
        return Spinorfield{Bout,Tout,M,typeof(Uout),u.ND,typeof(bufs[1])}(
            Uout, bufs..., sizeU..., u.NV, 3, u.topology
        )
    else
        return Fieldtype{Bout,Tout,M,typeof(Uout),typeof(bufs[1])}(
            Uout, bufs..., sizeU..., u.NV, 3, u.topology
        )
    end
end

# XXX: might not be needed
Base.eltype(u::AbstractField) = eltype(u.U)
Base.elsize(u::AbstractField) = Base.elsize(u.U)
Base.parent(u::AbstractField) = u.U
Base.pointer(u::AbstractField) = pointer(u.U)
Base.strides(u::AbstractField) = strides(u.U)
# This converts u to a PtrArray pointing to the entries of u.U, meaning that we cant
# access any of the fields of u within the @batch loop
@inline object_and_preserve(u::AbstractField) = object_and_preserve(u.U)
float_type(::AbstractArray{SMatrix{3,3,Complex{T},9},5}) where {T} = T
float_type(::AbstractField{B,T}) where {B,T} = T
KA.get_backend(u::AbstractField) = get_backend(u.U)

# define dims() function twice --- once for generic arrays, such that GPUs and @batch
# can use it, and once for Abstractfields for any other case
@inline dims(u) = NTuple{4,Int64}((size(u, 2), size(u, 3), size(u, 4), size(u, 5)))
@inline dims(u::AbstractField) = dims(u.U)
@inline dimrange(u, μ) = axes(u, μ+1)
@inline dimrange(u::AbstractField, μ) = axes(u.U, μ+1)
@inline global_dims(u::AbstractField) = u.topology.global_dims
@inline local_dims(u::AbstractField) = u.topology.local_dims
@inline local_ranges(u::AbstractField) = u.topology.local_ranges
Base.ndims(u::AbstractField) = 4
Base.size(u::AbstractField) = NTuple{5,Int64}((4, u.NX, u.NY, u.NZ, u.NT))

"""
    check_dims(x1, rest...)

Check if all fields have the same dimensions. Throw an `AssertionError` otherwise.
"""
@generated function check_dims(x1, rest::Vararg{Any,N}) where {N}
    q_inner = Expr(:comparison, :(global_dims(x1)))

    for i in 1:N
        push!(q_inner.args, :(==))
        push!(q_inner.args, :(global_dims(rest[$i])))
    end

    q = Expr(:macrocall, Symbol("@assert"), :(), q_inner)
    return q
end

@inline function Base.eachindex(u::AbstractField, fields...)
    check_dims(u, fields...)
    return eachindex(u)
end

@inline function Base.eachindex(arg::Union{Symbol,Bool}, u::AbstractField, fields...)
    check_dims(u, fields...)
    return eachindex(arg, u)
end

@inline Base.eachindex(u::AbstractField) = CartesianIndices((u.NX, u.NY, u.NZ, u.NT))
@inline Base.eachindex(::IndexLinear, u::AbstractField) = Base.OneTo(u.NV)
# For MPI parallelized fields:
@inline Base.eachindex(u::AbstractMPIField) = u.topology.bulk_sites
Base.eachindex(::IndexLinear, u::AbstractMPIField) =
    error("MPI parallelized field can not be iterated over linearly")

@inline function Base.eachindex(parity::Symbol, u::AbstractField)
    return u.topology.bulk_sites_eo[parity]
end

@inline function Base.eachindex(even::Bool, u::AbstractField)
    NX, NY, NZ, NT = global_dims(u)
    @assert iseven(NT)
    last_range = even ? (1:div(NT, 2)) : (div(NT, 2)+1:NT)
    return CartesianIndices((NX, NY, NZ, last_range))
end

@inline function Base.eachindex(even::Bool, u::AbstractMPIField)
    hw = u.topology.halo_width
    NT = global_dims(u)[4]
    @assert iseven(NT)
    irange = even ? (1+hw:div(NT, 2)) : (div(NT, 2)+1:NT-hw)
    return u.topology.bulk_sites[irange]
end

@inline allindices(u::AbstractField) = eachindex(IndexCartesian(), u.U) # all indices including halo regions

Base.length(u::AbstractField) = u.NV

# overload get and set for the Abstractfields structs, so we dont have to do u.U[μ,x,y,z,t]:
Base.@propagate_inbounds Base.getindex(u::AbstractField, i::Integer) = u.U[i]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μ, x, y, z, t) = u.U[μ, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μsite) = u.U[μsite]
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, i::Integer) =
    setindex!(u.U, v, i)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μ, x, y, z, t) =
    setindex!(u.U, v, μ, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μsite) =
    setindex!(u.U, v, μsite)

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", u::T) where {T<:AbstractField}
    print(io, "$(typeof(u))", "(;\n")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            println(io, "\t", fieldname, " = ", getfield(u, fieldname)[], ",")
        elseif fieldname == :topology
            println(io, "\t", fieldname, " = FieldTopology(...)")
        else
            println(io, "\t", fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

function Base.show(io::IO, u::T) where {T<:AbstractField}
    print(io, "$(typeof(u))", "(;\n")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            println(io, "\t", fieldname, " = ", getfield(u, fieldname)[], ",")
        elseif fieldname == :topology
            println(io, "\t", fieldname, " = FieldTopology(...)")
        else
            println(io, "\t", fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

end
