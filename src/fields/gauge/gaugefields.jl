module Gaugefields
	using Base.Threads: nthreads, threadid, @threads
	using LinearAlgebra
	using StaticArrays
	using Polyester
	using ..Utils

	abstract type Abstractfield end
	abstract type AbstractGaugeAction end

	struct Gaugefield{T<:AbstractGaugeAction} <: Abstractfield
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

		function Gaugefield(NX, NY, NZ, NT, β; kind_of_gaction = "wilson")
			U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

			for μ = 1:4
				Uμ = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
				U[μ] = Uμ
			end

			NV = NX * NY * NZ * NT

			if kind_of_gaction == "wilson"
				GaugeAction = WilsonGaugeAction()
			elseif kind_of_gaction == "symanzik_tree"
				GaugeAction = SymanzikTreeGaugeAction()
			elseif kind_of_gaction == "symanzik_tadpole"
				GaugeAction = SymanzikTadGaugeAction()
			elseif kind_of_gaction == "iwasaki"
				GaugeAction = IwasakiGaugeAction()
			elseif kind_of_gaction == "dbw2"
				GaugeAction = DBW2GaugeAction()
			else
				error("Gauge action '$(kind_of_gaction)' not supported")
			end

			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new{typeof(GaugeAction)}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
		end

		function Gaugefield(NX, NY, NZ, NT, β, GaugeAction)
			U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

			for μ = 1:4
				Uμ = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
				U[μ] = Uμ
			end

			NV = NX * NY * NZ * NT

			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new{typeof(GaugeAction)}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
		end

		function Gaugefield(u::Gaugefield{T}) where {T}
			NX, NY, NZ, NT = size(u)
			β = u.β
			U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

			for μ = 1:4
				Uμ = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
				U[μ] = Uμ
			end

			NV = NX * NY * NZ * NT

			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new{T}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
		end
	end

	struct TemporaryField <: Abstractfield
		U::Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64

		function TemporaryField(NX, NY, NZ, NT)
			U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

			for μ = 1:4
				Uμ = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
				U[μ] = Uμ
			end
			
			NV = NX * NY * NZ * NT
			NC = 3
			return new(U, NX, NY, NZ, NT, NV, NC)
		end

		function TemporaryField(u::Abstractfield)
			NX, NY, NZ, NT = size(u)
			return TemporaryField(NX, NY, NZ, NT)
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
				Uμ = Array{exp_iQ_su3, 4}(undef, NX, NY, NZ, NT)
				U[μ] = Uμ
			end
			
			NV = NX * NY * NZ * NT
			return new(U, NX, NY, NZ, NT, NV)
		end

		function CoeffField(u::Abstractfield)
			NX, NY, NZ, NT = size(u)
			return CoeffField(NX, NY, NZ, NT)
		end
	end

	function Base.setindex!(u::Abstractfield, v, μ)
        @inbounds u.U[μ] = v
		return nothing
    end

	@inline function Base.getindex(u::Abstractfield, μ)
        @inbounds return u.U[μ]
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

	function Base.size(u::Abstractfield)
        return (u.NX, u.NY, u.NZ, u.NT)
    end

	function Base.similar(u::T) where {T <: Abstractfield}
		if typeof(u) == TemporaryField
			uout = T(u)
		elseif typeof(u) <: Gaugefield
			uout = Gaugefield(u)
		elseif typeof(u) == CoeffField
			uout = T(u)
		end

		return uout
	end

	function Base.eltype(::Gaugefield{T}) where {T}
		return T
	end

	function substitute_U!(a::T, b::T) where {T <: Abstractfield}
		NX, NY, NZ, NT = size(a)

		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							a[μ][ix,iy,iz,it] = b[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end

		return nothing 
	end

	function swap_U!(a::Gaugefield, b::Gaugefield)
		NX, NY, NZ, NT = size(a)
		a.Sg, b.Sg = b.Sg, a.Sg
		a.CV, b.CV = b.CV, a.CV

		for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							a[μ][ix,iy,iz,it], b[μ][ix,iy,iz,it] =
								b[μ][ix,iy,iz,it], a[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end

		return nothing 
	end

	function identity_gauges(NX, NY, NZ, NT, β; gaction = "wilson")
		u = Gaugefield(NX, NY, NZ, NT, β, kind_of_gaction = gaction)
		
		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							u[μ][ix,iy,iz,it] = eye3
						end
					end
				end
			end
		end

        return u
    end

    function random_gauges(NX, NY, NZ, NT, β; gaction = "wilson")
		u = Gaugefield(NX, NY, NZ, NT, β, kind_of_gaction = gaction)

		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							link = @SMatrix rand(ComplexF64, 3, 3)
							link = proj_onto_SU3(link)
							u[μ][ix,iy,iz,it] = link
						end
					end
				end
			end
		end

		Sg = calc_gauge_action(u)
		u.Sg = Sg
        return u
    end

	function clear_U!(u::Abstractfield)
		NX, NY, NZ, NT = size(u)

		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds u[μ][ix,iy,iz,it] = zero(u[μ][ix,iy,iz,it])
						end
					end
				end
			end
		end

		return nothing
	end

	function normalize!(u::Gaugefield{<:AbstractGaugeAction})
		NX, NY, NZ, NT = size(u)

 		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							link = u[μ][ix,iy,iz,it]
							u[μ][ix,iy,iz,it] = proj_onto_SU3(link)
						end
					end
				end
			end
        end

        return nothing
    end

	function add!(a::Abstractfield, b::Abstractfield, fac)
		# there is never a case where one would want to add fields of different sizes
		@assert size(a) == size(b) "sizes of fields aren't the same"
		NX, NY, NZ, NT = size(a)
	
		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							a[μ][ix,iy,iz,it] += fac * b[μ][ix,iy,iz,it] 
						end
					end
				end
			end
		end
	
		return nothing
	end

	function LinearAlgebra.lmul!(tA, a::Abstractfield, b::Abstractfield; fac = 1)
		@assert size(a) == size(b) "sizes of fields aren't the same"
		NX, NY, NZ, NT = size(a)
	
		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							a[μ][ix,iy,iz,it] = 
								fac * tA(b[μ][ix,iy,iz,it]) * a[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end
	
		return nothing
	end

	function LinearAlgebra.rmul!(tA, a::Abstractfield, b::Abstractfield; fac = 1)
		@assert size(a) == size(b) "sizes of fields aren't the same"
		NX, NY, NZ, NT = size(a)
	
		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							a[μ][ix,iy,iz,it] = 
								fac * a[μ][ix,iy,iz,it] * tA(b[μ][ix,iy,iz,it])
						end
					end
				end
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
