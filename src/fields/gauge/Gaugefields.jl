module Gaugefields

using Base.Threads: nthreads, threadid, @threads
using LinearAlgebra
using StaticArrays
using Polyester
using Random
using ..Utils

abstract type Abstractfield end
abstract type AbstractGaugeAction end

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
struct Gaugefield{GA} <: Abstractfield
	# Vector of 4D-Arrays performs better than 5D-Array for some reason
	U::Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
	NC::Int64

	β::Float64
	Sg::Base.RefValue{Float64}
	CV::Base.RefValue{Float64}

	function Gaugefield(NX, NY, NZ, NT, β; GA=WilsonGaugeAction)
		U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

		for μ in 1:4
			U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
			fill!(U[μ], zero3)
		end

		NV = NX * NY * NZ * NT

		Sg = Base.RefValue{Float64}(0.0)
		CV = Base.RefValue{Float64}(0.0)
		return new{GA}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
	end
end

Gaugefield(U::Gaugefield{GA}) where {GA} = Gaugefield(U.NX, U.NY, U.NZ, U.NT, U.β; GA=GA)

"""
Temporaryfield(NX, NY, NZ, NT)
Temporaryfield(u::Abstractfield)

Creates a Temporaryfield, i.e. an array of 3-by-3 matrices of size `4 × NX × NY × NZ × NT`
or of the same size as `u`
"""
struct Temporaryfield <: Abstractfield
	U::Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
	NC::Int64

	function Temporaryfield(NX, NY, NZ, NT)
		U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

		for μ in 1:4
			U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
			fill!(U[μ], zero3)
		end

		NV = NX * NY * NZ * NT
		NC = 3
		return new(U, NX, NY, NZ, NT, NV, NC)
	end
end

Temporaryfield(u::Abstractfield) = Temporaryfield(u.NX, u.NY, u.NZ, u.NT)

"""
CoeffField(NX, NY, NZ, NT)
CoeffField(u::Abstractfield)

Creates a CoeffField, i.e. an array of `exp_iQ_su3` objects of size `4 × NX × NY × NZ × NT`
or of the same size as `u`. The objects hold the `Q`-matrices and all the exponential
parameters needed for stout-force recursion
"""
struct CoeffField <: Abstractfield
	U::Vector{Array{exp_iQ_su3, 4}}
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64

	function CoeffField(NX, NY, NZ, NT)
		U = Vector{Array{exp_iQ_su3, 4}}(undef, 4)

		for μ in 1:4
			U[μ] = Array{exp_iQ_su3, 4}(undef, NX, NY, NZ, NT)
			fill!(U[μ], exp_iQ_su3())
		end

		NV = NX * NY * NZ * NT
		return new(U, NX, NY, NZ, NT, NV)
	end
end

CoeffField(u::Abstractfield) = CoeffField(u.NX, u.NY, u.NZ, u.NT)

Base.size(u::Abstractfield) = NTuple{4, Int64}((u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(u::Abstractfield) = CartesianIndices(size(u))
Base.eachindex(::IndexLinear, u::Abstractfield) = Base.OneTo(u.NV)
Base.eltype(::Gaugefield{GA}) where {GA} = GA

function Base.setindex!(u::Abstractfield, v, μ)
	u.U[μ] = v
	return nothing
end

@inline function Base.getindex(u::Abstractfield, μ)
	return u.U[μ]
end

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
		setfield!(u, p, val)
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

function substitute_U!(a::T, b::T) where {T <: Abstractfield}
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = b[μ][site]
		end
	end

	return nothing
end

function initial_gauges(initial, args...; type_of_gaction=WilsonGaugeAction)
	if initial == "cold"
		return identity_gauges(args..., type_of_gaction)
	elseif initial == "hot"
		return random_gauges(args..., type_of_gaction)
	else
		error("Only cold or hot inital configs supported")
	end
end


function identity_gauges(NX, NY, NZ, NT, β, type_of_gaction)
	u = Gaugefield(NX, NY, NZ, NT, β, GA=type_of_gaction)

	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] = eye3
		end
	end

	return u
end

function random_gauges(NX, NY, NZ, NT, β, type_of_gaction)
	u = Gaugefield(NX, NY, NZ, NT, β, GA=type_of_gaction)

	for site = eachindex(u)
		for μ in 1:4
			link = @SMatrix rand(ComplexF64, 3, 3)
			link = proj_onto_SU3(link)
			u[μ][site] = link
		end
	end

	Sg = calc_gauge_action(u)
	u.Sg = Sg
	return u
end

function clear_U!(u::Abstractfield)
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] = zero3
		end
	end

	return nothing
end

function normalize!(u::Gaugefield)
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] = proj_onto_SU3(u[μ][site])
		end
	end

	return nothing
end

function add!(a::Abstractfield, b::Abstractfield, fac)
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] += fac * b[μ][site]
		end
	end

	return nothing
end

function mul!(u::Abstractfield, α::Number)
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] *= α
		end
	end

	return nothing
end

function leftmul!(a::Abstractfield, b::Abstractfield)
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = cmatmul_oo(b[μ][site], a[μ][site])
		end
	end

	return nothing
end

function leftmul_dagg!(a::Abstractfield, b::Abstractfield)
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = cmatmul_do(b[μ][site], a[μ][site])
		end
	end

	return nothing
end

function rightmul!(a::Abstractfield, b::Abstractfield)
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = cmatmul_oo(a[μ][site], b[μ][site])
		end
	end

	return nothing
end

function rightmul_dagg!(a::Abstractfield, b::Abstractfield)
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = cmatmul_od(a[μ][site], b[μ][site])
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

end
