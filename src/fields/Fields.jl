module Fields

using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester # Used for the @batch macro, which enables multi threading
using Random
using StaticArrays # Used for the SU3 matrices
using ..Utils # Contains utility functions, such as projections and the exponential map

import KernelAbstractions as KA # With this we can write generic GPU kernels for ROC and CUDA
import StrideArraysCore: object_and_preserve # This is used to convert the Abstractfield to a PtrArray in the @batch loop

# When CUDA.jl or AMDGPU.jl are loaded, their backends are appended to this Dict
const BACKEND = Dict{String,Type{<:Backend}}("cpu" => CPU)

# We are going to need these if we want to transfer a field from one backend to another
# For other backends, we overload this method in their respective extensions
array_type(::Type{CPU}) = Array

# Define an abstract field super type that is parametrized by the backend, the precision and
# the array type (Array, CuArray, ROCArray)
abstract type Abstractfield{B,T,A} end

include("gaugefields.jl")
include("liefields.jl")
include("fieldstrength.jl")
include("fermionfields.jl")
include("iterators.jl")
include("gpu_iterators.jl")
include("gpu_kernels/utils.jl")

include("field_operations.jl")
include("wilsonloops.jl")
include("actions.jl")
include("staples.jl")
include("clovers.jl")

include("gpu_kernels/field_operations.jl")
include("gpu_kernels/wilsonloops.jl")
include("gpu_kernels/actions.jl")
include("gpu_kernels/liefields.jl")
include("gpu_kernels/fieldstrength.jl")
include("gpu_kernels/fermionfields.jl")

Base.similar(u::Gaugefield) = Gaugefield(u)
Base.similar(u::Colorfield) = Colorfield(u)
Base.similar(u::Expfield) = Expfield(u)

"""
	to_backend(Bout, u::Abstractfield{Bin,T})

Ports the Abstractfield u to the backend Bout, maintaining all elements
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
function to_backend(::Type{Bout}, u::Abstractfield{Bin,T}) where {Bout,Bin,T}
    Bout === Bin && return u # no need to do anything if the backends are the same
    A = array_type(Bout)
    sizeU = dims(u)
    Uout = A(u.U)

    if u isa Gaugefield
        GA = gauge_action(u)
        Sg = Base.RefValue{Float64}(u.Sg)
        CV = Base.RefValue{Float64}(u.CV)
        return Gaugefield{Bout,T,typeof(Uout),GA}(Uout, sizeU..., u.NV, 3, u.β, Sg, CV)
    elseif u isa Expfield
        return Expfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Colorfield
        return Colorfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Tensorfield
        return Tensorfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Fermionfield
        return Fermionfield{Bout,T,typeof(Uout),u.ND}(Uout, sizeU..., u.NV, 3)
    else
        throw(ArgumentError("Unsupported field type"))
    end
end

# XXX: might not be needed
Base.eltype(u::Abstractfield) = eltype(u.U)
Base.elsize(u::Abstractfield) = Base.elsize(u.U)
Base.parent(u::Abstractfield) = u.U
Base.pointer(u::Abstractfield) = pointer(u.U)
Base.strides(u::Abstractfield) = strides(u.U)
# This converts u to a PtrArray pointing to the entries of u.U, meaning that we cant
# access any of the fields of u within the @batch loop
@inline object_and_preserve(u::Abstractfield) = object_and_preserve(u.U)
float_type(::AbstractArray{SMatrix{3,3,Complex{T},9},5}) where {T} = T
float_type(::Abstractfield{BACKEND,T}) where {BACKEND,T} = T
KA.get_backend(u::Abstractfield) = get_backend(u.U)

# define dims() function twice --- once for generic arrays, such that GPUs and @batch
# can use it, and once for Abstractfields for any other case
@inline dims(u) = NTuple{4,Int64}((size(u, 2), size(u, 3), size(u, 4), size(u, 5)))
@inline dims(u::Abstractfield) = NTuple{4,Int64}((u.NX, u.NY, u.NZ, u.NT))
@inline volume(u) = prod(dims(u))
@inline volume(u::Abstractfield) = u.NV
Base.ndims(u::Abstractfield) = 4
Base.size(u::Abstractfield) = NTuple{5,Int64}((4, u.NX, u.NY, u.NZ, u.NT))

"""
    check_dims(x1, rest...)

Check if all fields have the same dimensions. Throw an `AssertionError` otherwise.
"""
@generated function check_dims(x1, rest::Vararg{Any,N}) where {N}
    q_inner = Expr(:comparison, :(dims(x1)))
    for i in 1:N
        push!(q_inner.args, :(==))
        push!(q_inner.args, :(dims(rest[$i])))
    end
    q = Expr(:macrocall, Symbol("@assert"), :(), q_inner)
    return q
end

Base.eachindex(u::Abstractfield) = CartesianIndices((u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(::IndexLinear, u::Abstractfield) = Base.OneTo(u.NV)
function Base.eachindex(even::Bool, u::Abstractfield)
    NX, NY, NZ, NT = dims(u)
    @assert iseven(NT)
    last_range = even ? (1:div(NT, 2)) : (div(NT, 2)+1:NT)
    return CartesianIndices((NX, NY, NZ, last_range))
end
Base.length(u::Abstractfield) = u.NV

# overload get and set for the Abstractfields structs, so we dont have to do u.U[μ,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, x, y, z, t) = u.U[μ, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, x, y, z, t) =
    setindex!(u.U, v, μ, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", u::T) where {T<:Abstractfield}
    print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            print(io, " ", fieldname, " = ", getfield(u, fieldname)[], ",")
        else
            print(io, " ", fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

function Base.show(io::IO, u::T) where {T<:Abstractfield}
    print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            print(io, " ", fieldname, " = ", getfield(u, fieldname)[], ",")
        else
            print(io, " ", fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

end
