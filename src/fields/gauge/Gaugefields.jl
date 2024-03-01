module Gaugefields

using AMDGPU: ROCArray, ROCBackend
using CUDA: CuArray, CUDABackend, i32
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester
using StaticArrays
using Random
using ..Utils

import KernelAbstractions as KA
import StrideArraysCore: object_and_preserve
export @latmap, @latsum

const SUPPORTED_BACKENDS = (CPU, CUDABackend, ROCBackend)

array_type(::CPU) = Array
array_type(::CUDABackend) = CuArray
array_type(::ROCBackend) = ROCArray

abstract type Abstractfield{BACKEND,T,A} <: DenseArray{SU{3,9,T},5} end

Base.eltype(u::Abstractfield) = eltype(u.U)
Base.elsize(u::Abstractfield) = Base.elsize(u.U)
Base.parent(u::Abstractfield) = u.U
Base.pointer(u::Abstractfield) = pointer(u.U)
Base.strides(u::Abstractfield) = strides(u.U)
@inline object_and_preserve(u::Abstractfield) = object_and_preserve(u.U)
float_type(::AbstractArray{SMatrix{3,3,Complex{T},9},5}) where {T} = T
float_type(::Abstractfield{BACKEND,T}) where {BACKEND,T} = T
KA.get_backend(u::Abstractfield) = get_backend(u.U)

function Base.show(io::IO, ::MIME"text/plain", u::T) where {T<:Abstractfield}
	print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(T)
		fieldname ∈ (:U, :NV) && continue

		if fieldname ∈ (:Sg, :CV)
        	print(io, " ", fieldname, " = ", getfield(u, fieldname)[], ",")
		else
			print(io, " ", fieldname, " = ", getfield(u, fieldname), ",")
		end
    end
    print(io, ")")
	return nothing
end

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
struct Gaugefield{BACKEND,T,A<:AbstractArray{SU{3,9,T},5},GA} <: Abstractfield{BACKEND,T,A}
	U::A
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
	NC::Int64

	β::Float64
	Sg::Base.RefValue{Float64}
	CV::Base.RefValue{Float64}
end

function Gaugefield(NX, NY, NZ, NT, β; BACKEND=CPU, T=Float64, GA=WilsonGaugeAction)
	@assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
	U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
	NV = NX * NY * NZ * NT
	Sg = Base.RefValue{Float64}(0.0)
	CV = Base.RefValue{Float64}(0.0)
	return Gaugefield{BACKEND,T,typeof(U),GA}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
end

function Gaugefield(u::Gaugefield{BACKEND,T,A,GA}) where {BACKEND,T,A,GA}
	NX, NY, NZ, NT = dims(u)
	U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
	Sg = Base.RefValue{Float64}(0.0)
	CV = Base.RefValue{Float64}(0.0)
	return Gaugefield{BACKEND,T,typeof(U),GA}(U, NX, NY, NZ, NT, u.NV, u.NC, u.β, Sg, CV)
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
struct Temporaryfield{BACKEND,T,A<:AbstractArray{SU{3,9,T},5}} <: Abstractfield{BACKEND,T,A}
	U::A
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
	NC::Int64
end

function Temporaryfield(NX, NY, NZ, NT; BACKEND=CPU, T=Float64)
	@assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
	U = KA.zeros(BACKEND(), SMatrix{3,3,Complex{T},9}, 4, NX, NY, NZ, NT)
	NV = NX * NY * NZ * NT
	NC = 3
	return Temporaryfield{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV, NC)
end

function Temporaryfield(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
	NX, NY, NZ, NT = dims(u)
	U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
	return Temporaryfield{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, u.NV, u.NC)
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
struct CoeffField{BACKEND,T,A<:AbstractArray{exp_iQ_su3{T},5}} <: Abstractfield{BACKEND,T,A}
	U::A # TODO: Add support for arbitrary NC
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
end

function CoeffField(NX, NY, NZ, NT; BACKEND=CPU, T=Float64)
	@assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
	U = KA.zeros(BACKEND(), exp_iQ_su3{T}, 4, NX, NY, NZ, NT)
	NV = NX * NY * NZ * NT
	return CoeffField{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV)
end

function CoeffField(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
	NX, NY, NZ, NT = dims(u)
	U = KA.zeros(BACKEND(), exp_iQ_su3{T}, 4, NX, NY, NZ, NT)
	return CoeffField{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, u.NV)
end

# define dims() function twice --- once for generic arrays, such that CUDA and ROC
# can use it, and once for Abstractfields, such that CPU can use it
@inline dims(u) = NTuple{4,Int64}((size(u,2), size(u,3), size(u,4), size(u,5)))
@inline dims(u::Abstractfield) = NTuple{4,Int64}((u.NX, u.NY, u.NZ, u.NT))
Base.ndims(u::Abstractfield) = 4
Base.size(u::Abstractfield) = NTuple{5,Int64}((4, u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(u::Abstractfield) = CartesianIndices((u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(::IndexLinear, u::Abstractfield) = Base.OneTo(u.NV)
Base.length(u::Abstractfield) = u.NV
gactionT(::Gaugefield{D,T,A,GA}) where {D,T,A,GA} = GA

# overload get and set for the Abstractfields structs, so we dont have to do u.U[μ,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, x, y, z, t) = u.U[μ,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, site::SiteCoords) = u.U[μ,site]
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

function Base.similar(u::T) where {T<:Abstractfield} # creates a zero(u) unlike Base.similar
	if T <: Gaugefield
		uout = Gaugefield(u)
	else
		uout = T(u)
	end

	return uout
end

"""
	to_backend(Bout, u::Abstractfield{Bin,T})

Ports the Abstractfield u to the backend Bout, maintaining all link variables
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
function to_backend(::Type{Bout}, u::Abstractfield{Bin,T}) where {Bout,Bin,T}
	A = array_type(Bout())
	Uout = A(u.U)
	sizeU = dims(u)

	if u isa Gaugefield
		GA = gactionT(u)
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

function substitute_U!(a::Abstractfield{CPU,T}, b::Abstractfield{CPU,T}) where {T}
	@assert dims(a) == dims(b)

	@batch for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = b[μ,site]
		end
	end

	return nothing
end

function initial_gauges(initial, args...; BACKEND=CPU, T=Float64, GA=WilsonGaugeAction)
	@assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
	u = Gaugefield(args...; BACKEND=BACKEND, T=T, GA=GA)

	if initial == "cold"
		identity_gauges!(u)
	elseif initial === "hot"
		random_gauges!(u)
	else
		throw(AssertionError("Only COLD or HOT initial configs supported"))
	end

	return u
end

function identity_gauges!(u::Gaugefield{CPU,T}) where {T}
	@batch for site in eachindex(u)
		for μ in 1:4
			u[μ,site] = eye3(T)
		end
	end

	return nothing
end

function random_gauges!(u::Gaugefield{CPU,T}) where {T}
	for site = eachindex(u)
		for μ in 1:4
			link = @SMatrix rand(Complex{T}, 3, 3)
			link = proj_onto_SU3(link)
			u[μ,site] = link
		end
	end

	Sg = calc_gauge_action(u)
	u.Sg = Sg
	return nothing
end

function clear_U!(u::Gaugefield{CPU,T}) where {T} # set all link variables to zero
	@batch for site in eachindex(u)
		for μ in 1:4
			u[μ,site] = zero3(T)
		end
	end

	return nothing
end

function normalize!(u::Gaugefield{CPU})
	@batch for site in eachindex(u)
		for μ in 1:4
			u[μ,site] = proj_onto_SU3(u[μ,site])
		end
	end

	return nothing
end

function add!(a::Abstractfield{CPU,T}, b::Abstractfield{CPU}, fac) where {T}
	@assert dims(a) == dims(b)
	fac = T(fac)

	@batch for site in eachindex(a)
		for μ in 1:4
			a[μ,site] += fac * b[μ,site]
		end
	end

	return nothing
end

function mul!(u::Gaugefield{CPU,T}, α::Number) where {T}
	α = T(α)

	@batch for site in eachindex(u)
		for μ in 1:4
			u[μ,site] *= α
		end
	end

	return nothing
end

function leftmul!(a::Abstractfield{CPU}, b::Abstractfield{CPU})
	@assert dims(a) == dims(b)

	@batch for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_oo(b[μ,site], a[μ,site])
		end
	end

	return nothing
end

function leftmul_dagg!(a::Abstractfield{CPU}, b::Abstractfield{CPU})
	@assert dims(a) == dims(b)

	@batch for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_do(b[μ,site], a[μ,site])
		end
	end

	return nothing
end

function rightmul!(a::Abstractfield{CPU}, b::Abstractfield{CPU})
	@assert dims(a) == dims(b)

	@batch for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_oo(a[μ,site], b[μ,site])
		end
	end

	return nothing
end

function rightmul_dagg!(a::Abstractfield{CPU,T}, b::Abstractfield{CPU,T}) where {T}
	@assert dims(a) == dims(b)

	@batch for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_od(a[μ,site], b[μ,site])
		end
	end

	return nothing
end

include("../iterators.jl")
include("../gpu_iterators.jl")

include("wilsonloops.jl")
include("actions.jl")
include("staples.jl")
include("clovers.jl")
include("fieldstrength.jl")
include("liefields.jl")

include("gpu_kernels/utils.jl")
include("gpu_kernels/field_operations.jl")
include("gpu_kernels/wilsonloops.jl")
include("gpu_kernels/actions.jl")
include("gpu_kernels/liefields.jl")
include("gpu_kernels/fieldstrength.jl")

end
