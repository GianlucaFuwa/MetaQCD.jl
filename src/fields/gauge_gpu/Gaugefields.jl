module Gaugefields

using Base.Threads: nthreads, threadid, @threads
using CUDA
using LinearAlgebra
using KernelAbstractions
using StaticArrays
using Polyester
using Random
using ..Utils

abstract type Abstractfield{TI,TF,TA} end
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
struct Gaugefield{TI,TF,TA,GA} <: Abstractfield{TI,TF,TA}
	U::TA
	NX::TI
	NY::TI
	NZ::TI
	NT::TI
	NV::TI
	NC::TI

	β::TF
	Sg::Base.RefValue{TF}
	CV::Base.RefValue{TF}

	function Gaugefield(NX, NY, NZ, NT, β; GA=WilsonGaugeAction, device=CPU,
						TI=device==CPU ? Int64 : Int32, TF=device==CPU ? Float64 : Float32)
		if device == CPU
			U = Vector{Array{SMatrix{3, 3, Complex{TF}, 9}, 4}}(undef, 4)

			for μ in 1:4
				U[μ] = Array{SMatrix{3, 3, Complex{TF}, 9}, 4}(undef, NX, NY, NZ, NT)
				fill!(U[μ], zero3)
			end
		elseif device == GPU
			U = CuArray{SMatrix{3, 3, Complex{TF}, 9}, 5}(undef, 4, NX, NY, NZ, NT)
			fill!(U, zero3)
		end

		NV = NX * NY * NZ * NT

		Sg = Base.RefValue{TF}(0.0)
		CV = Base.RefValue{TF}(0.0)
		return new{TI,TF,typeof(U),GA}(U, TI(NX), TI(NY), TI(NZ), TI(NT), TI(NV), TI(NC),
										TF(β), Sg, CV)
	end
end

function Gaugefield(U::Gaugefield{TI,TF,TA,GA}) where {TI,TF,TA,GA}
	device = TA<:Vector ? CPU : GPU
	return Gaugefield(U.NX, U.NY, U.NZ, U.NT, U.β; GA=GA, device=device, TI=TI, TF=TF)
end

"""
	Temporaryfield(NX, NY, NZ, NT)
	Temporaryfield(u::Abstractfield)

Creates a Temporaryfield, i.e. an array of 3-by-3 matrices of size `4 × NX × NY × NZ × NT`
or of the same size as `u`
"""
struct Temporaryfield{TI,TF,TA} <: Abstractfield{TI,TF,TA}
	U::TA
	NX::TI
	NY::TI
	NZ::TI
	NT::TI
	NV::TI
	NC::TI

	function Temporaryfield(NX, NY, NZ, NT; device=CPU, TI=device==CPU ? Int64 : Int32,
							TF=device==CPU ? Float64 : Float32)
		if device == CPU
			U = Vector{Array{SMatrix{3, 3, Complex{TF}, 9}, 4}}(undef, 4)

			for μ in 1:4
				U[μ] = Array{SMatrix{3, 3, Complex{TF}, 9}, 4}(undef, NX, NY, NZ, NT)
				fill!(U[μ], zero3)
			end
		elseif device == GPU
			U = CuArray{SMatrix{3, 3, Complex{TF}, 9}, 5}(undef, 4, NX, NY, NZ, NT)
			fill!(U, zero3)
		end

		NV = NX * NY * NZ * NT
		NC = 3
		return new{TI,TF,typeof(U)}(U, TI(NX), TI(NY), TI(NZ), TI(NT), TI(NV), TI(NC))
	end
end

function Temporaryfield(U::Abstractfield{TI,TF,TA}) where {TI,TF,TA}
	device = TA<:Vector ? CPU : GPU
	return Temporaryfield(U.NX, U.NY, U.NZ, U.NT; device=device, TI=TI, TF=TF)
end

"""
	CoeffField(NX, NY, NZ, NT)
	CoeffField(u::Abstractfield)

Creates a CoeffField, i.e. an array of `exp_iQ_su3` objects of size `4 × NX × NY × NZ × NT`
or of the same size as `u`. The objects hold the `Q`-matrices and all the exponential
parameters needed for stout-force recursion
"""
struct CoeffField{TI,TF,TA} <: Abstractfield{TI,TF,TA}
	U::Vector{Array{exp_iQ_su3, 4}}
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64

	function CoeffField(NX, NY, NZ, NT; device=CPU, TI=device==CPU ? Int64 : Int32,
						TF=device==CPU ? Float64 : Float32)
		if device == CPU
			U = Vector{Array{SMatrix{3, 3, exp_iQ_su3{TF}, 9}, 4}}(undef, 4)

			for μ in 1:4
				U[μ] = Array{SMatrix{3, 3, exp_iQ_su3{TF}, 9}, 4}(undef, NX, NY, NZ, NT)
				fill!(U[μ], zero3)
			end
		elseif device == GPU
			U = CuArray{SMatrix{3, 3, exp_iQ_su3{TF}, 9}, 5}(undef, 4, NX, NY, NZ, NT)
			fill!(U, zero3)
		end

		NV = NX * NY * NZ * NT
		return new{TI,TF,typeof(U)}(U, TI(NX), TI(NY), TI(NZ), TI(NT), TI(NV), TI(NC))
	end
end

function CoeffField(U::Abstractfield{TI,TF,TA}) where {TI,TF,TA}
	device = TA<:Vector ? CPU : GPU
	return CoeffField(U.NX, U.NY, U.NZ, U.NT; device=device, TI=TI, TF=TF)
end

function Base.size(u::Abstractfield{TI,TF,TA}) where {TI,TF,TA}
	return NTuple{4, TI}((u.NX, u.NY, u.NZ, u.NT))
end

function Base.eachindex(u::Abstractfield{TI,TF,TA}) where {TI,TF,TA}
	return CartesianIndices(size(u)::NTuple{4, TI})
end

Base.eachindex(::IndexLinear, u::T)  where {T<:Abstractfield} = Base.OneTo(u.NV)
Base.eltype(::Gaugefield{TI,TF,TA,GA}) where {TI,TF,TA,GA} = GA

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

	return nothing
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
