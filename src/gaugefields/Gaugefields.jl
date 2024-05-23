module Gaugefields

using AMDGPU: ROCArray, ROCBackend
using CUDA
using CUDA: i32
using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester # Used for the @batch macro, which enables multi threading
using StaticArrays # Used for the SU3 matrices
using StaticArrays: check_dims
using Random
using ..Utils # Contains utility functions, such as projections and the exponential map

import KernelAbstractions as KA # With this we can write generic GPU kernels for ROC and CUDA
import StrideArraysCore: object_and_preserve # This is used to convert the Abstractfield to a PtrArray in the @batch loop

const SUPPORTED_BACKENDS = (CPU, CUDABackend, ROCBackend)
const BACKEND = Dict{String,Type{<:Backend}}(
    "cpu" => CPU, "cuda" => CUDABackend, "roc" => ROCBackend
)

# We are going to need these if we want to transfer a field from one backend to another
# see `to_backend` function
array_type(::Type{CPU}) = Array
array_type(::Type{CUDABackend}) = CuArray
array_type(::Type{ROCBackend}) = ROCArray

# Define an abstract field super type that is parametrized by the backend, the precision and
# the array type (Array, CuArray, ROCArray). Make it a subtype of DenseArray so that
# @batch knows how to handle it
abstract type Abstractfield{BACKEND,T,A} end

"""
	Gaugefield(NX, NY, NZ, NT, β; BACKEND=CPU, T=Float64, GA=WilsonGaugeAction)
	Gaugefield(U::Gaugefield)

Creates a Gaugefield on `BACKEND`, i.e. an array of link-variables (SU3 matrices with `T`
precision) of size `4 × NX × NY × NZ × NT` with coupling parameter `β` and gauge action `GA`
or a zero-initialized copy of `U`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
# Supported gauge actions
`WilsonGaugeAction` \\
`SymanzikTreeGaugeAction` (Lüscher-Weisz) \\
`IwasakiGaugeAction` \\
`DBW2GaugeAction`
"""
struct Gaugefield{BACKEND,T,A,GA} <: Abstractfield{BACKEND,T,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    β::Float64 # Seems weird to have it here, but I couldnt be bothered passing it as an argument everywhere
    Sg::Base.RefValue{Float64} # Current Gauge action, used to safe work
    CV::Base.RefValue{Float64} # Current collective variable, used to safe work
    function Gaugefield(NX, NY, NZ, NT, β; BACKEND=CPU, T=Float64, GA=WilsonGaugeAction)
        @assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{BACKEND,T,typeof(U),GA}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
    end
end

function Gaugefield(parameters)
    NX, NY, NZ, NT = parameters.L
    β = parameters.beta
    GA = GAUGE_ACTION[parameters.gauge_action]
    T = FLOAT_TYPE[parameters.float_type]
    B = BACKEND[parameters.backend]
    U = Gaugefield(NX, NY, NZ, NT, β; BACKEND=B, T=T, GA=GA)

    initial = parameters.initial
    if initial == "cold"
        identity_gauges!(U)
    elseif initial == "hot"
        random_gauges!(U)
    else
        error("intial condition \"$(initial)\" not supported, only \"cold\" or \"hot\"")
    end

    return U
end

function Gaugefield(u::Gaugefield{BACKEND,T,A,GA}) where {BACKEND,T,A,GA}
    return Gaugefield(u.NX, u.NY, u.NZ, u.NT, u.β; BACKEND=BACKEND, T=T, GA=GA)
end

"""
	Temporaryfield(NX, NY, NZ, NT; backend=CPU(), T=Val(Float64))
	Temporaryfield(u::Abstractfield)

Creates a Temporaryfield on `backend`, i.e. an array of 3-by-3 `T`-precision matrices of
size `4 × NX × NY × NZ × NT` or a zero-initialized Temporaryfield of the same size as `u`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Temporaryfield{BACKEND,T,A} <: Abstractfield{BACKEND,T,A}
    U::A
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    NC::Int64
    function Temporaryfield(NX, NY, NZ, NT; BACKEND=CPU, T=Float64)
        @assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        NC = 3
        return new{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV, NC)
    end
end

function Temporaryfield(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
    return Temporaryfield(u.NX, u.NY, u.NZ, u.NT; BACKEND=BACKEND, T=T)
end

"""
	CoeffField(NX, NY, NZ, NT; backend=CPU(), T=Val(Float64))
	CoeffField(u::Abstractfield)

Creates a CoeffField on `backend`, i.e. an array of `T`-precison `exp_iQ_su3` objects of
size `4 × NX × NY × NZ × NT` or of the same size as `u`. The objects hold the `Q`-matrices
and all the exponential parameters needed for stout-force recursion
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct CoeffField{BACKEND,T,A} <: Abstractfield{BACKEND,T,A}
    U::A # TODO: Add support for arbitrary NC
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    function CoeffField(NX, NY, NZ, NT; BACKEND=CPU, T=Float64)
        @assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
        U = KA.zeros(BACKEND(), exp_iQ_su3{T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        return new{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV)
    end
end

function CoeffField(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
    return CoeffField(u.NX, u.NY, u.NZ, u.NT; BACKEND=BACKEND, T=T)
end

# overload some function such that @batch knows how to handle Abstractfields
Base.eltype(u::Abstractfield) = eltype(u.U)
Base.elsize(u::Abstractfield) = Base.elsize(u.U)
Base.parent(u::Abstractfield) = u.U
Base.pointer(u::Abstractfield) = pointer(u.U)
Base.strides(u::Abstractfield) = strides(u.U)
# This converts u to a PtrArray pointing the entries of u.U, meaning that we shouldnt access
# any of the fields of u within the @batch loop
@inline object_and_preserve(u::Abstractfield) = object_and_preserve(u.U)
float_type(::AbstractArray{SMatrix{3,3,Complex{T},9},5}) where {T} = T
float_type(::Abstractfield{BACKEND,T}) where {BACKEND,T} = T
KA.get_backend(u::Abstractfield) = get_backend(u.U)

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

# define dims() function twice --- once for generic arrays, such that CUDA and ROC
# can use it, and once for Abstractfields, such that CPU can use it
@inline dims(u) = NTuple{4,Int64}((size(u, 2), size(u, 3), size(u, 4), size(u, 5)))
@inline dims(u::Abstractfield{CPU}) = NTuple{4,Int64}((u.NX, u.NY, u.NZ, u.NT))
@inline volume(u::Abstractfield) = u.NV
Base.ndims(u::Abstractfield) = 4
Base.size(u::Abstractfield) = NTuple{5,Int64}((4, u.NX, u.NY, u.NZ, u.NT))

function Gaugefields.check_dims(u, rest...)
    @nospecialize u rest
    udims = dims(u)

    for field in rest
        @assert dims(field) == udims
    end
    return nothing
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

gauge_action(::Gaugefield{B,T,A,GA}) where {B,T,A,GA} = GA

# overload get and set for the Abstractfields structs, so we dont have to do u.U[μ,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, x, y, z, t) = u.U[μ, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, x, y, z, t) =
    setindex!(u.U, v, μ, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)

# overload getproperty and setproperty!, because we dont want to deal with Base.RefValues
# all the time
function Base.getproperty(u::Gaugefield, p::Symbol)
    if p == :Sg
        return getfield(u, :Sg)[]
    elseif p == :CV
        return getfield(u, :CV)[]
    else
        return getfield(u, p)
    end
end

function Base.setproperty!(u::Gaugefield, p::Symbol, val)
    if p == :Sg
        getfield(u, :Sg)[] = val
    elseif p == :CV
        getfield(u, :CV)[] = val
    else
        setproperty!(u, p, val)
    end

    return nothing
end

Base.similar(u::Gaugefield) = Gaugefield(u)
Base.similar(u::Temporaryfield) = Temporaryfield(u)
Base.similar(u::CoeffField) = CoeffField(u)

"""
	to_backend(Bout, u::Abstractfield{Bin,T})

Ports the Abstractfield u to the backend Bout, maintaining all link variables
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
    elseif u isa CoeffField
        return CoeffField{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Temporaryfield
        return Temporaryfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Tensorfield
        return Tensorfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    else
        throw(ArgumentError("Unsupported field type"))
    end
end

include("iterators.jl")
include("gpu_iterators.jl")
include("gpu_kernels/utils.jl")

include("field_operations.jl")
include("wilsonloops.jl")
include("actions.jl")
include("staples.jl")
include("clovers.jl")
include("liefields.jl")
include("fieldstrength.jl")
include("fermionfields.jl")

include("gpu_kernels/field_operations.jl")
include("gpu_kernels/wilsonloops.jl")
include("gpu_kernels/actions.jl")
include("gpu_kernels/liefields.jl")
include("gpu_kernels/fieldstrength.jl")

# Need to add this function to CUDA, because the base implementation is dynamic
CUDA.@device_override @noinline function Base.__throw_rational_argerror_typemin(
    ::Type{T}
) where {T}
    CUDA.@print_and_throw "invalid rational: denominator can't be typemin"
end

end
