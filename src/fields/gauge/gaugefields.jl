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
        @inbounds u.U[μ] = v
		return nothing
    end

	@inline function Base.getindex(u::T, μ) where {T <: Abstractfield}
        @inbounds return u.U[μ]
    end

    function Base.eachindex(u::T) where {T <: Abstractfield}
        return Base.OneTo(u.NV)
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
		@assert size(a) == size(b) "swapped fields need to be of same size"
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							a[μ][ix,iy,iz,it] = b[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end

		return nothing
	end

	function identity_gauges(NX, NY, NZ, NT, β; type_of_gaction = WilsonGaugeAction)
		u = Gaugefield(NX, NY, NZ, NT, β, GA = type_of_gaction)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							u[μ][ix,iy,iz,it] = eye3
						end
					end
				end
			end
		end

        return u
    end

    function random_gauges(NX, NY, NZ, NT, β; type_of_gaction = WilsonGaugeAction)
		u = Gaugefield(NX, NY, NZ, NT, β, GA = type_of_gaction)
		# No multithreading to make random elements reproducible
		# (could also use static scheduler but then we'd have to use @threads,
		# which doesn't default to non-multithreaded for 1 thread)
		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
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

	function clear_U!(u::T) where {T <: Abstractfield}
		NX, NY, NZ, NT = size(u)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							u[μ][ix,iy,iz,it] = zero3
						end
					end
				end
			end
		end

		return nothing
	end

	function normalize!(u::T) where {T <: Gaugefield}
		NX, NY, NZ, NT = size(u)

 		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							u[μ][ix,iy,iz,it] = proj_onto_SU3(u[μ][ix,iy,iz,it])
						end
					end
				end
			end
        end

        return nothing
    end

	function add!(a::T1, b::T2, fac) where {T1, T2 <: Abstractfield}
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							a[μ][ix,iy,iz,it] += fac * b[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end

		return nothing
	end

	function leftmul!(a::T1, b::T2) where {T1 <: Abstractfield, T2 <: Abstractfield}
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							a[μ][ix,iy,iz,it] =
                                cmatmul_oo(b[μ][ix,iy,iz,it], a[μ][ix,iy,iz,it])
						end
					end
				end
			end
		end

		return nothing
	end

    function leftmul_dagg!(a::T1, b::T2) where {T1 <: Abstractfield, T2 <: Abstractfield}
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							a[μ][ix,iy,iz,it] =
                                cmatmul_do(b[μ][ix,iy,iz,it], a[μ][ix,iy,iz,it])
						end
					end
				end
			end
		end

		return nothing
	end

	function rightmul!(a::T1, b::T2) where {T1 <: Abstractfield, T2 <: Abstractfield}
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							a[μ][ix,iy,iz,it] =
                                cmatmul_oo(a[μ][ix,iy,iz,it], b[μ][ix,iy,iz,it])
						end
					end
				end
			end
		end

		return nothing
	end

    function rightmul_dagg!(a::T1, b::T2) where {T1 <: Abstractfield, T2 <: Abstractfield}
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						@inbounds for μ in 1:4
							a[μ][ix,iy,iz,it] =
                                cmatmul_od(a[μ][ix,iy,iz,it], b[μ][ix,iy,iz,it])
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
