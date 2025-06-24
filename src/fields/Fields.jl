module Fields

using Base.Meta: quot
using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using MacroTools
using OffsetArrays
using Polyester # Used for the @batch macro, which enables multi threading
using Random
using StaticArrays # Used for the SU3 matrices
using ..Utils # Contains utility functions, such as projections and the exponential map

import Adapt: adapt_structure
import KernelAbstractions as KA # With this we can write generic GPU kernels for ROC and CUDA
import StrideArraysCore: PtrArray, object_and_preserve # This is used to convert the AbstractField to a PtrArray in the @batch loop

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

include("constructor.jl")
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
include("adapt.jl")

include("action.jl") # Gauge action methods
include("field_operations.jl") # General operations on fields, like adding, copying etc.
include("stencils/plaquette.jl") # Definition of clover operator
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

# XXX: Not sure why these are here, but whatever
Base.similar(u::Gaugefield) = Gaugefield(u)
Base.similar(u::Colorfield) = Colorfield(u)
Base.similar(u::Expfield) = Expfield(u)
Base.similar(u::Tensorfield) = Tensorfield(u)
Base.similar(u::Spinorfield) = Spinorfield(u)
Base.similar(u::MultiSpinorfield) = MultiSpinorfield(u)

# Base.view(u::AbstractField, I::CartesianIndices{4}) = view(u.U, 1:4, I.indices...)
# Base.view(u::AbstractField, I::Vector{CartesianIndex{4}}) = view(u.U, 1:4, I)
# Base.view(u::Tensorfield, I::CartesianIndices{4}) = view(u.U, 1:4, 1:4, I.indices...)
# Base.view(u::Tensorfield, I::Vector{CartesianIndex{4}}) = view(u.U, 1:4, 1:4, I)
# Base.view(f::Spinorfield, I::CartesianIndices{4}) = view(f.U, I.indices...)
# Base.view(f::Spinorfield, I::Vector{CartesianIndex{4}}) = view(f.U, I)
# Base.view(f::MultiSpinorfield, s, I::CartesianIndices{4}) = view(f.U, s, I.indices...)
# Base.view(f::MultiSpinorfield, s, I::Vector{CartesianIndex{4}}) = view(f.U, s, I)

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

    Fieldtype = eval(nameof(typeof(u)))
    A = array_type(Bout)
    new_eltype = convert(Tout, eltype(u.U))
    Uout = A{new_eltype}(u.U)
    sendbuf = isnothing(u.sendbuf) ? nothing : A{new_eltype}(u.sendbuf)
    halos = if isnothing(u.halos)
        nothing
    else
        ntuple(Val(8)) do i
            OffsetArray(
                KA.zeros(Bout(), new_eltype, size(u.halos[i])),
                eachindex(u.halos[i])...
            )
        end
    end

    if u isa Gaugefield
        GA = gauge_action(u)
        return Gaugefield{Bout,Tout,M,GA}(Uout, sendbuf, halos, u.topology, u.β)
    elseif u isa Spinorfield
        ND = num_dirac(u)
        return Spinorfield{Bout,Tout,M,ND}(Uout, sendbuf, halos, u.topology)
    elseif u isa MultiSpinorfield
        ND = num_dirac(u)
        return MultiSpinorfield{Bout,Tout,M,ND}(Uout, sendbuf, halos, u.topology, u.numspinors)
    elseif u isa Paulifield
        C = has_clover_term(u)
        return Paulifield{Bout,Tout,M,C}(Uout, sendbuf, halos, u.topology, u.csw)
    else
        return Fieldtype{Bout,Tout,M}(Uout, sendbuf, halos, u.topology)
    end
end

# XXX: might not be needed
Base.eltype(u::AbstractField) = eltype(u.U)
Base.elsize(u::AbstractField) = Base.elsize(u.U)
Base.parent(u::AbstractField) = u.U
Base.pointer(u::AbstractField) = pointer(u.U)
Base.strides(u::AbstractField) = strides(u.U)

# Some useful functions that share geometry information about the fields
KA.get_backend(u::AbstractField) = get_backend(u.U)
KA.get_backend(u::OffsetArray) = get_backend(u.parent)
Base.length(u::AbstractField) = u.topology.global_volume
Base.size(u::AbstractField) = u.topology.global_dims
Base.size(u::AbstractField, μ) = u.topology.global_dims[μ]
Base.axes(u::AbstractField) = u.topology.bulk_sites_padded.indices
Base.axes(u::AbstractField, μ::Integer) = u.topology.bulk_sites_padded.indices[μ]
# Base.axes(u::AbstractField, μ::Integer) = u.topology.bulk_sites.indices[μ]
@inline float_type(::AbstractField{B,T}) where {B,T} = T
@inline num_colors(u::AbstractField) = 3
@inline get_local_dims(u::AbstractField) = u.topology.local_dims
@inline get_local_volume(u::AbstractField) = u.topology.local_volume
@inline get_global_volume(u::AbstractField) = u.topology.global_volume
@inline get_halo_width(u::AbstractField) = u.topology.halo_width
@inline get_numprocs_cart(u::AbstractField) = u.topology.numprocs_cart

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
    check_dims(u, fields...)
    return allindices(u)
end

@inline allindices(u::AbstractField) = eachindex(IndexCartesian(), u.U) # all indices including halo regions

# overload get and set for the Abstractfields structs, so we dont have to do u.U[μ,x,y,z,t]:
Base.@propagate_inbounds Base.getindex(u::AbstractField, i::Integer) = u.U[i]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μ, x, y, z, t) = u.U[μ, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.getindex(u::AbstractField, μsite) = u.U[μsite]

Base.@propagate_inbounds function Base.getindex(u::AbstractMPIField, μ, site::SiteCoords)
    site in u.topology.bulk_sites && return u.U[μ, site]
    ihalo = get_halo_index(site, u.topology.bulk_sites)
    # return try
    #     u.halos[ihalo][μ, site]
    # catch _
    #     @error("$(mpi_myrank()), $site, $ihalo, $(u.topology.bulk_sites)")
    # end
    return u.halos[ihalo][μ, site]
end

Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, i::Integer) =
    setindex!(u.U, v, i)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μ, x, y, z, t) =
    setindex!(u.U, v, μ, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)
Base.@propagate_inbounds Base.setindex!(u::AbstractField, v, μsite) =
    setindex!(u.U, v, μsite)

Base.@propagate_inbounds function Base.setindex!(u::AbstractMPIField, v, μ, site::SiteCoords)
    bulk = u.topology.bulk_sites

    if site in bulk
        u.U[μ, site] = v
    else
        ihalo = get_halo_index(site, bulk)
        u.halos[ihalo][μ, site] = v
    end

    return nothing
end

function check_types(::Type{B}, ::Type{T}, U, halos, sendbuf) where {B,T}
    # some sanity checks
    # if !(U isa PtrArray)
    #     @assert get_backend(U) isa B
    # end
    #
    # if !isnothing(halos)
    #     for halo in halos
    #         if !(halo isa PtrArray)
    #             @assert get_backend(halo) isa B
    #         end
    #     end
    # end
    #
    # if !isnothing(sendbuf)
    #     if !(sendbuf isa PtrArray)
    #         @assert get_backend(sendbuf) isa B
    #     end
    # end

    @assert eltype(eltype(U)) === Complex{T}
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

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", u::T) where {T<:AbstractField}
    print(io, "$(nameof(typeof(u)))", "(\n")
    for fieldname in fieldnames(T)
        if fieldname in (:U, :halos, :sendbuf)
            println(io, "\t", fieldname, ": $(nameof(typeof(getfield(u, fieldname))))")
        elseif fieldname == :topology
            println(io, "\t", fieldname, ": FieldTopology")
        else
            println(io, "\t", fieldname, " = ", getfield(u, fieldname))
        end
    end
    print(io, ")")
    return nothing
end

function Base.show(io::IO, u::T) where {T<:AbstractField}
    print(io, "$(nameof(typeof(u)))", "(\n")
    for fieldname in fieldnames(T)
        if fieldname in (:U, :halos, :sendbuf)
            println(io, "\t", fieldname, " = $(nameof(typeof(getfield(u, fieldname))))()")
        elseif fieldname == :topology
            println(io, "\t", fieldname, " = FieldTopology(...)")
        else
            println(io, "\t", fieldname, " = ", getfield(u, fieldname))
        end
    end
    print(io, ")")
    return nothing
end

end
