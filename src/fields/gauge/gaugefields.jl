module Gaugefields
	using Base.Threads: nthreads, threadid, @threads
	using LinearAlgebra
	using StaticArrays
	using Polyester
	using ..Utils

	abstract type Abstractfield end
	abstract type AbstractGaugeAction end

	struct Gaugefield{TG} <: Abstractfield
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

		function Gaugefield(NX, NY, NZ, NT, β; TG = WilsonGaugeAction)
			U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)

			for μ = 1:4
				Uμ = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
				U[μ] = Uμ
			end

			NV = NX * NY * NZ * NT

			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new{TG}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
		end

		function Gaugefield(u::Gaugefield{TG}) where {TG}
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
			return new{TG}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
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
				Uμ = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
				U[μ] = Uμ
			end
			
			NV = NX * NY * NZ * NT
			NC = 3
			return new(U, NX, NY, NZ, NT, NV, NC)
		end

		function Temporaryfield(u::Abstractfield)
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
			setfield!(u, p, val)
		end

		return nothing
	end

	function Base.size(u::Abstractfield)
        return (u.NX, u.NY, u.NZ, u.NT)
    end

	function Base.similar(u::T) where {T <: Abstractfield}
		if T <: Gaugefield
			uout = Gaugefield(u)
		else
			uout = T(u)
		end

		return uout
	end

	function Base.eltype(::Gaugefield{TG}) where {TG}
		return TG
	end

	function substitute_U!(a::T, b::T) where {T <: Abstractfield}
		@assert size(a) == size(b) "swapped fields need to be of same size"
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds a[μ][ix,iy,iz,it] = b[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end

		return nothing 
	end

	function swap_U!(a::T, b::T) where {T<:Gaugefield}
		@assert size(a) == size(b) "swapped fields need to be of same size"
		NX, NY, NZ, NT = size(a)
		a.Sg, b.Sg = b.Sg, a.Sg
		a.CV, b.CV = b.CV, a.CV

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds a[μ][ix,iy,iz,it], b[μ][ix,iy,iz,it] =
								b[μ][ix,iy,iz,it], a[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end

		return nothing 
	end

	function identity_gauges(NX, NY, NZ, NT, β; type_of_gaction = WilsonGaugeAction)
		u = Gaugefield(NX, NY, NZ, NT, β, TG = type_of_gaction)
		
		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds u[μ][ix,iy,iz,it] = eye3
						end
					end
				end
			end
		end

        return u
    end

    function random_gauges(NX, NY, NZ, NT, β; type_of_gaction = WilsonGaugeAction)
		u = Gaugefield(NX, NY, NZ, NT, β, TG = type_of_gaction)
		# No multithreading to make random elements reproducible
		# (could also use static scheduler but then we'd have to use @threads,
		# which doesn't default to non-multithreaded for 1 thread)
		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							link = @SMatrix rand(ComplexF64, 3, 3)
							link = proj_onto_SU3(link)
							@inbounds u[μ][ix,iy,iz,it] = link
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

		@batch for it in 1:NT
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

	function normalize!(u::T) where {T<:Gaugefield}
		NX, NY, NZ, NT = size(u)

 		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds u[μ][ix,iy,iz,it] = proj_onto_SU3(u[μ][ix,iy,iz,it])
						end
					end
				end
			end
        end

        return nothing
    end

	function add!(a::Abstractfield, b::Abstractfield, fac)
		@assert size(a) == size(b) "added fields need to be of same size"
		NX, NY, NZ, NT = size(a)
	
		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds a[μ][ix,iy,iz,it] += fac * b[μ][ix,iy,iz,it] 
						end
					end
				end
			end
		end
	
		return nothing
	end

	function leftmul!(tA, a::Abstractfield, b::Abstractfield; fac = 1)
		@assert size(a) == size(b) "multed fields need to be of same size"
		NX, NY, NZ, NT = size(a)
	
		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds a[μ][ix,iy,iz,it] = 
								fac * tA(b[μ][ix,iy,iz,it]) * a[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end
	
		return nothing
	end

	function rightmul!(tA, a::Abstractfield, b::Abstractfield; fac = 1)
		@assert size(a) == size(b) "swapped fields need to be of same size"
		NX, NY, NZ, NT = size(a)
	
		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds a[μ][ix,iy,iz,it] = 
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
