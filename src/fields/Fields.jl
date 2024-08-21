module Fields

using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using MPI
using Polyester # Used for the @batch macro, which enables multi threading
using Random
using StaticArrays # Used for the SU3 matrices
using ..Utils # Contains utility functions, such as projections and the exponential map

import KernelAbstractions as KA # With this we can write generic GPU kernels for ROC and CUDA
import StrideArraysCore: object_and_preserve # This is used to convert the AbstractField to a PtrArray in the @batch loop

const COMM = MPI.COMM_WORLD
const MYRANK = MPI.Comm_rank(COMM)
const COMM_SIZE = MPI.Comm_size(COMM)

# When CUDA.jl or AMDGPU.jl are loaded, their backends are appended to this Dict
const BACKENDS = Dict{String,Type{<:KA.Backend}}("cpu" => CPU)

# We are going to need these if we want to transfer a field from one backend to another
# For other backends, we overload this method in their respective extensions
array_type(::Type{CPU}) = Array

# Define an abstract field super type that is parametrized by the backend, the precision and
# the array type (Array, CuArray, ROCArray)
abstract type AbstractField{Backend,FloatType,MPIParallel,ArrayType} end
const AbstractMPIField{B,T,A} = AbstractField{B,T,true,A}

include("distributed.jl")
include("gaugefield.jl")
include("algebrafield.jl")
include("spinorfield.jl")
include("tensorfield.jl")
include("iterators.jl")
include("gpu_iterators.jl")
include("gpu_kernels/utils.jl")

include("action.jl")
include("field_operations.jl")
include("clover.jl")
include("staple.jl")
include("wilsonloop.jl")

include("gpu_kernels/action.jl")
include("gpu_kernels/algebrafield.jl")
include("gpu_kernels/field_operations.jl")
include("gpu_kernels/spinorfield.jl")
include("gpu_kernels/tensorfield.jl")
include("gpu_kernels/wilsonloop.jl")

Base.similar(u::Gaugefield) = Gaugefield(u)
Base.similar(u::Colorfield) = Colorfield(u)
Base.similar(u::Expfield) = Expfield(u)

"""
	to_backend(Backend_out, u::AbstractField{Backend_in,FloatType})

Ports the AbstractField u to the backend `Backend_out`, maintaining all elements
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
function to_backend(::Type{Bout}, u::AbstractField{M,Bin,T}) where {M,Bout,Bin,T}
    @assert M === false "Switching backends not yet supported with MPI parallelization" # FIXME

    if Bout === Bin
        u_out = similar(u)
        copy!(u_out, u)
        return u_out
    end

    A = array_type(Bout)
    sizeU = global_dims(u)
    Uout = A(u.U)

    if u isa Gaugefield
        GA = gauge_action(u)
        Sg = Base.RefValue{Float64}(u.Sg)
        CV = Base.RefValue{Float64}(u.CV)
        return Gaugefield{M,Bout,T,typeof(Uout),GA}(Uout, sizeU..., u.NV, 3, u.β, Sg, CV)
    elseif u isa Expfield
        return Expfield{M,Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Colorfield
        return Colorfield{M,Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Tensorfield
        return Tensorfield{M,Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Fermionfield
        return Fermionfield{M,Bout,T,typeof(Uout),u.ND}(Uout, sizeU..., u.NV, 3)
    else
        throw(ArgumentError("Unsupported field type"))
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
float_type(::AbstractField{M,B,T}) where {M,B,T} = T
KA.get_backend(u::AbstractField) = get_backend(u.U)

# define dims() function twice --- once for generic arrays, such that GPUs and @batch
# can use it, and once for Abstractfields for any other case
@inline dims(u) = NTuple{4,Int64}((size(u, 2), size(u, 3), size(u, 4), size(u, 5)))
@inline dims(u::AbstractField) = NTuple{4,Int64}((size(u.U, 2), size(u.U, 3), size(u.U, 4), size(u.U, 5)))
@inline global_dims(u::AbstractField) = NTuple{4,Int64}((u.NX, u.NY, u.NZ, u.NT))
@inline local_dims(u) = NTuple{4,Int64}((size(u, 2), size(u, 3), size(u, 4), size(u, 5)))
@inline local_dims(u::AbstractField) = NTuple{4,Int64}((u.my_NX, u.my_NY, u.my_NZ, u.my_NT))
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

Base.eachindex(u::AbstractField) = CartesianIndices((u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(::IndexLinear, u::AbstractField) = Base.OneTo(u.NV)
# For MPI parallelized fields:
@inline function Base.eachindex(u::AbstractMPIField)
    pad = u.pad
    st = 1 + pad
    out = CartesianIndices((st:u.my_NX+pad, st:u.my_NY+pad, st:u.my_NZ+pad, st:u.my_NT+pad))
    return out
end

Base.eachindex(::IndexLinear, u::AbstractMPIField) =
    error("MPI parallelized field can not be iterated over linearly")

function Base.eachindex(even::Bool, u::AbstractField)
    NX, NY, NZ, NT = global_dims(u)
    @assert iseven(NT)
    last_range = even ? (1:div(NT, 2)) : (div(NT, 2)+1:NT)
    return CartesianIndices((NX, NY, NZ, last_range))
end

Base.length(u::AbstractField) = u.NV

const ONE_SITE = CartesianIndex(1, 1, 1, 1)

# overload get and set for the Abstractfields structs, so we dont have to do u.U[μ,x,y,z,t]:
Base.@propagate_inbounds Base.getindex(u::AbstractField, μ, x, y, z, t) = u.U[μ, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μ, x, y, z, t) =
    setindex!(u.U, v, μ, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", u::T) where {T<:AbstractField}
    print(io, "$(typeof(u))", "(;\n")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            println(io, "\t", fieldname, " = ", getfield(u, fieldname)[], ",")
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
        else
            println(io, "\t", fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

end
