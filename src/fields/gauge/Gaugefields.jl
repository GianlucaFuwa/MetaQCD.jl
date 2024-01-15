module Gaugefields

using Base.Threads: nthreads, threadid, @threads
using LinearAlgebra
using StaticArrays
using Polyester
using Random
using ..Utils

abstract type Abstractfield end
abstract type AbstractGaugeAction end

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

Base.size(u::T) where {T<:Abstractfield} = NTuple{4, Int64}((u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(u::T) where {T<:Abstractfield} = CartesianIndices(size(u)::NTuple{4, Int64})
Base.eachindex(::IndexLinear, u::T)  where {T<:Abstractfield} = Base.OneTo(u.NV)
Base.eltype(::Gaugefield{GA}) where {GA} = GA

@inline function Base.setindex!(u::T, v, μ) where {T<:Abstractfield}
	u.U[μ] = v
	return nothing
end

function Base.getindex(u::T, μ) where {T<:Abstractfield}
	return u.U[μ]
end

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

function substitute_U!(a::Ta, b::Tb) where {Ta<:Abstractfield,Tb<:Abstractfield}
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = b[μ][site]
		end
	end

	return nothing
end

function initial_gauges(initial, args...; type_of_gaction=WilsonGaugeAction)
	u = Gaugefield(args...; GA=type_of_gaction)

	if initial == "cold"
		identity_gauges!(u)
	elseif initial == "hot"
		random_gauges!(u)
	else
		throw(AssertionError("Only cold or hot initial configs supported"))
	end

	return u
end

function identity_gauges!(u)
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] = eye3
		end
	end

	return nothing
end

function random_gauges!(u)
	for site = eachindex(u)
		for μ in 1:4
			link = @SMatrix rand(ComplexF64, 3, 3)
			link = proj_onto_SU3(link)
			u[μ][site] = link
		end
	end

	Sg = calc_gauge_action(u)
	u.Sg = Sg
	return nothing
end

function clear_U!(u::T) where {T<:Gaugefield}
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] = zero3
		end
	end

	return nothing
end

function normalize!(u::T) where {T<:Gaugefield}
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] = proj_onto_SU3(u[μ][site])
		end
	end

	return nothing
end

function add!(a::Ta, b::Tb, fac) where {Ta<:Abstractfield,Tb<:Abstractfield}
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] += fac * b[μ][site]
		end
	end

	return nothing
end

function mul!(u::T, α::Number) where {T<:Gaugefield}
	@batch per=thread for site in eachindex(u)
		for μ in 1:4
			u[μ][site] *= α
		end
	end

	return nothing
end

function leftmul!(a::Ta, b::Tb) where {Ta<:Abstractfield,Tb<:Abstractfield}
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = cmatmul_oo(b[μ][site], a[μ][site])
		end
	end

	return nothing
end

function leftmul_dagg!(a::Ta, b::Tb) where {Ta<:Abstractfield,Tb<:Abstractfield}
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = cmatmul_do(b[μ][site], a[μ][site])
		end
	end

	return nothing
end

function rightmul!(a::Ta, b::Tb) where {Ta<:Abstractfield,Tb<:Abstractfield}
	@batch per=thread for site in eachindex(a)
		for μ in 1:4
			a[μ][site] = cmatmul_oo(a[μ][site], b[μ][site])
		end
	end

	return nothing
end

function rightmul_dagg!(a::Ta, b::Tb) where {Ta<:Abstractfield,Tb<:Abstractfield}
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
