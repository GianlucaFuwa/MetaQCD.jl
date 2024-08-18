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

include("algebrafield.jl")
include("gaugefield.jl")
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
    sizeU = dims(u)
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
@inline dims(u::AbstractField) = NTuple{4,Int64}((u.NX, u.NY, u.NZ, u.NT))
@inline local_dims(u::AbstractField) = NTuple{4,Int64}((u.my_NX, u.my_NY, u.my_NZ, u.my_NT))
Base.ndims(u::AbstractField) = 4
Base.size(u::AbstractField) = NTuple{5,Int64}((4, u.NX, u.NY, u.NZ, u.NT))

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
    NX, NY, NZ, NT = dims(u)
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
    print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            println(io, fieldname, " = ", getfield(u, fieldname)[], ",")
        else
            println(io, fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

function Base.show(io::IO, u::T) where {T<:AbstractField}
    print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            println(io, fieldname, " = ", getfield(u, fieldname)[], ",")
        else
            println(io, fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

update_halo!(::AbstractField{B,T,false}) where {B,T} = nothing

function update_halo!(u::AbstractField{B,T,true}) where {B,T}
    comm_cart = u.comm_cart

    send_buf = u.halo_sendbuf
    recv_buf = u.halo_recvbuf
    requests = []

    for dir in 1:4
        prev_neighbor, next_neighbor = MPI.Cart_shift(comm_cart, dir-1, 1)
        # Source sites (bulk boundary):
        prev_sites_from, next_sites_from = bulk_boundary_sites(u, dir)
        # Sink sites (halo):
        prev_sites_to, next_sites_to = halo_sites(u, dir)

        dims = halo_dims(u, dir)

        # send - negative direction
        for site in prev_sites_from
            for μ in 1:4
                i = cartesian_to_linear(site, dims...)
                send_buf[μ, i] = u[μ, site]
            end
        end

        push!(requests, MPI.Isend(send_buf, comm_cart; dest=prev_neighbor))

        # send - positive direction
        for site in next_sites_from
            for μ in 1:4
                i = cartesian_to_linear(site, dims...)
                send_buf[μ, i] = u[μ, site]
            end
        end

        push!(requests, MPI.Isend(send_buf, comm_cart; dest=next_neighbor))

        # recv - negative direction
        push!(requests, MPI.Irecv!(recv_buf, comm_cart; source=next_neighbor))

        for site in prev_sites_to
            for μ in 1:4
                i = cartesian_to_linear(site, dims...)
                u[μ, site] = recv_buf[μ, i]
            end
        end

        # recv - positive direction
        push!(requests, MPI.Irecv!(recv_buf, comm_cart; source=prev_neighbor))

        for site in next_sites_to
            for μ in 1:4
                i = cartesian_to_linear(site, dims...)
                u[μ, site] = recv_buf[μ, i]
            end
        end
    end

    MPI.Waitall!(requests)
end

@inline function halo_dims(u::AbstractField, dir)
    my_NX, my_NY, my_NZ, my_NT = local_dims(u)
    pad = u.pad

    if dir == 1
        return (pad, my_NY, my_NZ, my_NT)
    elseif dir == 2
        return (my_NX, pad, my_NZ, my_NT)
    elseif dir == 3
        return (my_NX, my_NY, pad, my_NT)
    elseif dir == 4
        return (my_NX, my_NY, my_NZ, pad)
    else
        throw(AssertionError("dir has to be between 1 and 4"))
    end

    return dims
end

@inline function halo_sites(u::AbstractField, dim)
    my_NX, my_NY, my_NZ, my_NT = local_dims(u)
    pad = u.pad

    if dim == 1
        prev = CartesianIndices((1:pad, my_NY, my_NZ, my_NT))
        next = CartesianIndices((my_NX+1:my_NX+pad, my_NY, my_NZ, my_NT))
        return prev, next
    elseif dim == 2
        prev = CartesianIndices((my_NX, 1:pad, my_NZ, my_NT))
        next = CartesianIndices((my_NX, my_NY+1:my_NY+pad, my_NZ, my_NT))
        return prev, next
    elseif dim == 3
        prev = CartesianIndices((my_NX, my_NY, 1:pad, my_NT))
        next = CartesianIndices((my_NX, my_NY, my_NZ+1:my_NZ+pad, my_NT))
        return prev, next
    elseif dim == 4
        prev = CartesianIndices((my_NX, my_NY, my_NZ, 1:pad))
        next = CartesianIndices((my_NX, my_NY, my_NZ, my_NT+1:my_NT+pad))
        return prev, next
    else
        throw(AssertionError("dim has to be between 1 and 4"))
    end
end

@inline function bulk_boundary_sites(u::AbstractField, dim)
    my_NX, my_NY, my_NZ, my_NT = local_dims(u)
    pad = u.pad

    if dim == 1
        prev = CartesianIndices((1+pad:2pad, my_NY, my_NZ, my_NT))
        next = CartesianIndices((my_NX-pad:my_NX, my_NY, my_NZ, my_NT))
        return prev, next
    elseif dim == 2
        prev = CartesianIndices((my_NX, 1+pad:2pad, my_NZ, my_NT))
        next = CartesianIndices((my_NX, my_NY-pad:my_NY, my_NZ, my_NT))
        return prev, next
    elseif dim == 3
        prev = CartesianIndices((my_NX, my_NY, 1+pad:2pad, my_NT))
        next = CartesianIndices((my_NX, my_NY, my_NZ-pad:my_NZ, my_NT))
        return prev, next
    elseif dim == 4
        prev = CartesianIndices((my_NX, my_NY, my_NZ, 1+pad:2pad))
        next = CartesianIndices((my_NX, my_NY, my_NZ, my_NT-pad:my_NT))
        return prev, next
    else
        throw(AssertionError("dim has to be between 1 and 4"))
    end
end

end
