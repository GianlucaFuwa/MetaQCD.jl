module Gaugefields

using AMDGPU: ROCArray, ROCBackend
using Base.Threads: nthreads, threadid, @threads
using CUDA: CuArray, CUDABackend, i32
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using StaticArrays
using Polyester
using Random
using ..Utils

import KernelAbstractions as KA

struct CPUD end
struct GPUD end
abstract type Abstractfield{D,FloatT,A} end
abstract type AbstractGaugeAction end

floatT(::AbstractArray{SMatrix{3,3,Complex{FloatT},9},5}) where {FloatT} = FloatT
floatT(::Abstractfield{D,FloatT}) where {D,FloatT} = FloatT
KA.get_backend(u::Abstractfield) = get_backend(u.U)

function Base.show(io::IO, u::T) where {T<:Abstractfield}
	print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(typeof(u))
		fieldname∈(:U, :NV) && continue

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
	Gaugefield(NX, NY, NZ, NT, β; GA=WilsonGaugeAction)
	Gaugefield(U::Gaugefield{GA}) where {GA}

Creates a Gaugefield, i.e. an array of link-variables (SU3 matrices) of size
`4 × NX × NY × NZ × NT` with coupling parameter `β` and gauge action `GA` or a copy of `U`
# Supported gauge actions
`WilsonGaugeAction` \\
`SymanzikTreeGaugeAction` (Lüscher-Weisz) \\
`IwasakiGaugeAction` \\
`DBW2GaugeAction`
"""
struct Gaugefield{D,T,A<:AbstractArray{SU{3,9,T}, 5},GA} <: Abstractfield{D,T,A}
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

function Gaugefield(NX, NY, NZ, NT, β; backend=CPU(), T=Val(Float64),
	GA::AbstractGaugeAction=WilsonGaugeAction())
	D = if backend isa CPU
		CPUD
	elseif backend isa GPU
		GPUD
	else
		throw(AssertionError("Only CPU, CUDABackend or ROCBackend supported!"))
	end

	Tu = _unwrap_val(T)
	U = KA.zeros(backend, SU{3,9,Tu}, 4, NX, NY, NZ, NT)
	NV = NX * NY * NZ * NT
	Sg = Base.RefValue{Float64}(0.0)
	CV = Base.RefValue{Float64}(0.0)
	return Gaugefield{D,Tu,typeof(U),typeof(GA)}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
end

function Gaugefield(u::Gaugefield{D,T,A,GA}) where {D,T,A,GA}
	backend = get_backend(u)
	return Gaugefield(u.NX, u.NY, u.NZ, u.NT, u.β; backend=backend, T=Val(T), GA=GA())
end

"""
	Temporaryfield(NX, NY, NZ, NT)
	Temporaryfield(u::Abstractfield)

Creates a Temporaryfield, i.e. an array of 3-by-3 matrices of size `4 × NX × NY × NZ × NT`
or of the same size as `u`
"""
struct Temporaryfield{D,T,A<:AbstractArray{SU{3,9,T}, 5}} <: Abstractfield{D,T,A}
	U::A
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
	NC::Int64
end

function Temporaryfield(NX, NY, NZ, NT; backend=CPU(), T=Val(Float64))
	D = if backend isa CPU
		CPUD
	elseif backend isa GPU
		GPUD
	else
		throw(AssertionError("Only CPU, CUDABackend or ROCBackend supported!"))
	end

	Tu = _unwrap_val(T)
	U = KA.zeros(backend, SMatrix{3,3,Complex{Tu},9}, 4, NX, NY, NZ, NT)
	NV = NX * NY * NZ * NT
	NC = 3
	return Temporaryfield{D,Tu,typeof(U)}(U, NX, NY, NZ, NT, NV, NC)
end

function Temporaryfield(u::Abstractfield{D,T,A}) where {D,T,A}
	backend = get_backend(u)
	return Temporaryfield(u.NX, u.NY, u.NZ, u.NT; backend=backend, T=Val(T))
end

"""
	CoeffField(NX, NY, NZ, NT)
	CoeffField(u::Abstractfield)

Creates a CoeffField, i.e. an array of `exp_iQ_su3` objects of size `4 × NX × NY × NZ × NT`
or of the same size as `u`. The objects hold the `Q`-matrices and all the exponential
parameters needed for stout-force recursion
"""
struct CoeffField{D,T,A<:AbstractArray{exp_iQ_su3{T}, 5}} <: Abstractfield{D,T,A}
	U::A
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
end

function CoeffField(NX, NY, NZ, NT; backend=CPU(), T=Val(Float64))
	D = if backend isa CPU
		CPUD
	elseif backend isa GPU
		GPUD
	else
		throw(AssertionError("Only CPU, CUDABackend or ROCBackend supported!"))
	end

	Tu = _unwrap_val(T)
	U = KA.zeros(backend, exp_iQ_su3{Tu}, 4, NX, NY, NZ, NT)
	NV = NX * NY * NZ * NT
	return CoeffField{D,Tu,typeof(U)}(U, NX, NY, NZ, NT, NV)
end

function CoeffField(u::Abstractfield{D,T,A}) where {D,T,A}
	backend = get_backend(u)
	return CoeffField(u.NX, u.NY, u.NZ, u.NT; backend=backend, T=Val(T))
end

Base.ndims(u::T) where {T<:Abstractfield} = 4
Base.size(u::T) where {T<:Abstractfield} = NTuple{5, Int64}((4, u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(u::T) where {T<:Abstractfield} =
	CartesianIndices((u.NX, u.NY, u.NZ, u.NT)::NTuple{4, Int64})
Base.eachindex(::IndexLinear, u::T)  where {T<:Abstractfield} = Base.OneTo(u.NV)
Base.length(u::T) where {T<:Abstractfield} = u.NV
gactionT(::Gaugefield{D,T,A,GA}) where {D,T,A,GA} = GA

@inline function Base.setindex!(u::T, v, μ, x, y, z, t) where {T<:Abstractfield}
	u.U[μ,x,y,z,t] = v
	return nothing
end

@inline function Base.setindex!(u::T, v, μ, site::SiteCoords) where {T<:Abstractfield}
	u.U[μ,site] = v
	return nothing
end

Base.getindex(u::T, μ, x, y, z, t) where {T<:Abstractfield} = u.U[μ,x,y,z,t]
Base.getindex(u::T, μ, site::SiteCoords) where {T<:Abstractfield} = u.U[μ,site]

function Base.getproperty(u::T, p::Symbol) where {T<:Gaugefield}
	if p == :Sg
		return getfield(u, :Sg)[]
	elseif p == :CV
		return getfield(u, :CV)[]
	else
		return getfield(u, p)
	end
end

function Base.setproperty!(u::T, p::Symbol, val) where {T<:Gaugefield}
	if p == :Sg
		getfield(u, :Sg)[] = val
	elseif p == :CV
		getfield(u, :CV)[] = val
	else
		setproperty!(u, p, val)
	end

	return nothing
end

function Base.similar(u::T) where {T<:Abstractfield}
	if T <: Gaugefield
		uout = Gaugefield(u)
	else
		uout = T(u)
	end

	return uout
end

function to_backend(::Type{B}, u::Abstractfield{D,T}) where {B<:AbstractArray,D,T}
	Dnew = if B == Array
		CPUD
	elseif B == CuArray
		GPUD
	elseif B == ROCArray
		GPUD
	else
		throw(AssertionError("Only Array, CuArray or ROCArray supported!"))
	end

	Unew = B(u.U)
	sizeU = size(u)[2:end]
	if u isa Gaugefield
		return Gaugefield{Dnew,floatT(u),typeof(Unew),gactionT(u)}(Unew, sizeU..., u.NV,
		3, u.β, Base.RefValue{Float64}(u.Sg), Base.RefValue{Float64}(u.CV))
	elseif u isa CoeffField
		return CoeffField{Dnew,floatT(u),typeof(Unew)}(Unew, sizeU..., u.NV, 3)
	elseif u isa Temporaryfield
		return Temporaryfield{Dnew,floatT(u),typeof(Unew)}(Unew, sizeU..., u.NV, 3)
	else
		throw(AssertionError("Abstractfield type $T not supported!"))
	end
end

function substitute_U!(a::Abstractfield{CPUD,T}, b::Abstractfield{CPUD,T}) where {T}
	@assert size(a) == size(b)
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = b[μ,site]
		end
	end

	return nothing
end

function initial_gauges(initial, args...; backend=CPU(), T=Val(Float64),
	GA=WilsonGaugeAction())
	return _initial_gauges(initial, args..., backend, T, GA)
end

function _initial_gauges(initial, NX, NY, NZ, NT, β, ::CPU, ::Val{T}, GA) where {T}
	u = Gaugefield(NX, NY, NZ, NT, β; backend=CPU(), T=Val(T), GA=GA)

	if initial == "cold"
		identity_gauges!(u)
	elseif initial == "hot"
		random_gauges!(u)
	else
		throw(AssertionError("Only cold or hot initial configs supported"))
	end

	return u
end

function identity_gauges!(u::Gaugefield{CPUD,T}) where {T}
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ,site] = eye3(T)
		end
	end

	return nothing
end

function random_gauges!(u::Gaugefield{CPUD,T}) where {T}
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

function clear_U!(u::Gaugefield{CPUD,T}) where {T}
	@threads for site in eachindex(u)
		for μ in 1:4
			u[μ,site] = zero3(T)
		end
	end

	return nothing
end

function normalize!(u::Gaugefield{CPUD,T}) where {T}
	@threads for site in eachindex(u)
		for μ in 1:4
			u[μ,site] = proj_onto_SU3(u[μ,site])
		end
	end

	return nothing
end

function add!(a::Abstractfield{CPUD,T}, b::Abstractfield{CPUD,T}, fac) where {T}
	@assert size(a) == size(b)
	@threads for site in eachindex(a)
		for μ in 1:4
			a[μ,site] += fac * b[μ,site]
		end
	end

	return nothing
end

function mul!(u::Gaugefield{CPUD,T}, α::Number) where {T}
	@threads for site in eachindex(u)
		for μ in 1:4
			u[μ,site] *= α
		end
	end

	return nothing
end

function leftmul!(a::Abstractfield{CPUD,T}, b::Abstractfield{CPUD,T}) where {T}
	@assert size(a) == size(b)
	@threads for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_oo(b[μ,site], a[μ,site])
		end
	end

	return nothing
end

function leftmul_dagg!(a::Abstractfield{CPUD,T}, b::Abstractfield{CPUD,T}) where {T}
	@assert size(a) == size(b)
	@threads for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_do(b[μ,site], a[μ,site])
		end
	end

	return nothing
end

function rightmul!(a::Abstractfield{CPUD,T}, b::Abstractfield{CPUD,T}) where {T}
	@assert size(a) == size(b)
	@threads for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_oo(a[μ,site], b[μ,site])
		end
	end

	return nothing
end

function rightmul_dagg!(a::Abstractfield{CPUD,T}, b::Abstractfield{CPUD,T}) where {T}
	@assert size(a) == size(b)
	@threads for site in eachindex(a)
		for μ in 1:4
			a[μ,site] = cmatmul_od(a[μ,site], b[μ,site])
		end
	end

	return nothing
end

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
