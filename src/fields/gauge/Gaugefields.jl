"""
    Gaugefields

Module containing all subtypes of the abstract type 'Abstractfield' and their methods:

    Gaugefield{GA} -> struct holding all gauge links, the lattice size, the coupling;
                      keeps track of the current gauge action and CV for MetaD
                      and is parametrized by the kind of gauge action 'GA'
                      GA is in itself an empty struct that is used as a functor for action
                      and staple calculation

    Temporaryfield -> also defined on all gauge link sites, but only used for internally
                      used fields, like staples, forces etc.

    CoeffField -> holds matrix-exp. coefficients for use in stout-smearing (recursion)

    Liefield -> same as Temporaryfield with different name for verbosity, used in HMC
"""
module Gaugefields
	using Base.Threads: nthreads, threadid, @threads
	using LinearAlgebra
	using StaticArrays
	using Polyester
	using Random
	using ..Utils

	abstract type Abstractfield end
	abstract type AbstractGaugeAction end

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

		function Gaugefield(NX, NY, NZ, NT, β; GA = WilsonGaugeAction)
			U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

			for μ = 1:4
				U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
			end

			NV = NX * NY * NZ * NT

			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new{GA}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
		end

		function Gaugefield(u::Gaugefield{GA}) where {GA}
			NX, NY, NZ, NT = size(u)
			β = u.β
			U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

			for μ = 1:4
				U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
			end

			NV = NX * NY * NZ * NT

			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new{GA}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
		end
	end

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

			for μ = 1:4
				U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
			end

			NV = NX * NY * NZ * NT
			NC = 3
			return new(U, NX, NY, NZ, NT, NV, NC)
		end

		function Temporaryfield(u::T) where {T <: Abstractfield}
			NX, NY, NZ, NT = size(u)
			return Temporaryfield(NX, NY, NZ, NT)
		end
	end

	struct CoeffField <: Abstractfield
		U::Vector{Array{exp_iQ_su3, 4}}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64

		function CoeffField(NX, NY, NZ, NT)
			U = Vector{Array{exp_iQ_su3, 4}}(undef, 4)

			for μ = 1:4
				U[μ] = Array{exp_iQ_su3, 4}(undef, NX, NY, NZ, NT)
			end

			NV = NX * NY * NZ * NT
			return new(U, NX, NY, NZ, NT, NV)
		end

		function CoeffField(u::T) where {T <: Abstractfield}
			NX, NY, NZ, NT = size(u)
			return CoeffField(NX, NY, NZ, NT)
		end
	end

	function Base.setindex!(u::T, v, μ) where {T <: Abstractfield}
        u.U[μ] = v
		return nothing
    end

	@inline function Base.getindex(u::T, μ) where {T <: Abstractfield}
        return u.U[μ]
    end

	function Base.getproperty(u::T, p::Symbol) where {T <: Gaugefield}
		if p == :Sg
			return getfield(u, :Sg)[]
		elseif p == :CV
			return getfield(u, :CV)[]
		else
			return getfield(u, p)
		end
	end

	function Base.setproperty!(u::T, p::Symbol, val) where {T <: Gaugefield}
		if p == :Sg
			getfield(u, :Sg)[] = val
		elseif p == :CV
			getfield(u, :CV)[] = val
		else
			setfield!(u, p, val)
		end

		return nothing
	end

	function Base.size(u::Abstractfield)
        return (u.NX, u.NY, u.NZ, u.NT)
    end

    function Base.eachindex(u::Abstractfield)
        return CartesianIndices(size(u))
    end

    function Base.eachindex(::IndexLinear, u::Abstractfield)
        return Base.OneTo(u.NV)
    end

	function Base.similar(u::T) where {T <: Abstractfield}
		if T <: Gaugefield
			uout = Gaugefield(u)
		else
			uout = T(u)
		end

		return uout
	end

	function Base.eltype(::Gaugefield{GA}) where {GA}
		return GA
	end

	function substitute_U!(a::T, b::T) where {T <: Abstractfield}
		@batch for site in eachindex(a)
            for μ in 1:4
                a[μ][site] = b[μ][site]
            end
        end

		return nothing
	end

	function identity_gauges(NX, NY, NZ, NT, β; type_of_gaction = WilsonGaugeAction)
		u = Gaugefield(NX, NY, NZ, NT, β, GA = type_of_gaction)

		@batch for site in eachindex(u)
            for μ in 1:4
                u[μ][site] = eye3
            end
        end

        return u
    end

    function random_gauges(NX, NY, NZ, NT, β; type_of_gaction = WilsonGaugeAction)
		u = Gaugefield(NX, NY, NZ, NT, β, GA = type_of_gaction)

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

	function clear_U!(u::T) where {T <: Abstractfield}
		@batch for site in eachindex(u)
            for μ in 1:4
                u[μ][site] = zero3
            end
		end

		return nothing
	end

	function normalize!(u::T) where {T <: Gaugefield}
 		@batch for site in eachindex(u)
            for μ in 1:4
                u[μ][site] = proj_onto_SU3(u[μ][site])
            end
        end

        return nothing
    end

	function add!(a::Abstractfield, b::Abstractfield, fac)
		@batch for site in eachindex(a)
            for μ in 1:4
                a[μ][site] += fac * b[μ][site]
            end
		end

		return nothing
	end

	function leftmul!(a::Abstractfield, b::Abstractfield)
		@batch for site in eachindex(a)
            for μ in 1:4
                a[μ][site] = cmatmul_oo(b[μ][site], a[μ][site])
            end
		end

		return nothing
	end

    function leftmul_dagg!(a::Abstractfield, b::Abstractfield)
		@batch for site in eachindex(a)
            for μ in 1:4
                a[μ][site] = cmatmul_do(b[μ][site], a[μ][site])
            end
		end

		return nothing
	end

	function rightmul!(a::Abstractfield, b::Abstractfield)
		@batch for site in eachindex(a)
            for μ in 1:4
                a[μ][site] = cmatmul_oo(a[μ][site], b[μ][site])
            end
		end

		return nothing
	end

    function rightmul_dagg!(a::Abstractfield, b::Abstractfield)
		@batch for site in eachindex(a)
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
