module Gaugefields
	using Random
	using LinearAlgebra
	using StaticArrays
	using Polyester
	using TimerOutputs
	using ..Utils

	import Base.Threads: nthreads, threadid

	const too = TimerOutput()

	struct Gaugefield
		U::Vector{Array{SMatrix{3,3,ComplexF64,9},4}}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64

		β::Float64
		kind_of_gaction::String
		Sg::Base.RefValue{Float64}
		CV::Base.RefValue{Float64}

		function Gaugefield(NX, NY, NZ, NT, β; kind_of_gaction = "wilson")
			U = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 0)

			for μ = 1:4
				Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef, NX, NY, NZ, NT)
				fill!(Uμ, SMatrix{3,3,ComplexF64,9}(zeros(3, 3)))
				push!(U, Uμ)
			end

			NV = NX * NY * NZ * NT

			type_of_gaction = kind_of_gaction
			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new(U, NX, NY, NZ, NT, NV, 3, β, type_of_gaction, Sg, CV)
		end
	end

	struct TemporaryField
		U::Vector{Array{SMatrix{3,3,ComplexF64,9},4}}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64

		function TemporaryField(NX, NY, NZ, NT)
			U = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 0)

			for μ = 1:4
				Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef, NX, NY, NZ, NT)
				fill!(Uμ, SMatrix{3,3,ComplexF64,9}(zeros(3, 3)))
				push!(U, Uμ)
			end
			
			NV = NX * NY * NZ * NT
			NC = 3
			return new(U, NX, NY, NZ, NT, NV, NC)
		end

		function TemporaryField(U::Gaugefield)
			NX, NY, NZ, NT = size(U)
			return TemporaryField(NX, NY, NZ, NT)
		end
	end

	function Base.setindex!(U::Gaugefield, v, μ)
        @inbounds U.U[μ] = v
		return nothing
    end

	@inline function Base.getindex(U::Gaugefield, μ)
        @inbounds return U.U[μ]
    end

	function Base.setindex!(t::TemporaryField, v, μ)
		@inbounds t.U[μ] = v
		return nothing
	end

	@inline function Base.getindex(t::TemporaryField, μ)
		@inbounds return t.U[μ]
	end

	function Base.getproperty(U::Gaugefield, p::Symbol)
		if p == :Sg 
			return getfield(U, :Sg)[]
		elseif p == :CV
			return getfield(U, :CV)[]
		else 
			return getfield(U, p)
		end
	end

	function Base.setproperty!(U::Gaugefield, p::Symbol, val)
		if p == :Sg 
			getfield(U, :Sg)[] = val
		elseif p == :CV
			getfield(U, :CV)[] = val
		else 
			setfield!(U, p, val)
		end

		return nothing
	end

	function Base.size(U::Gaugefield)
        return U.NX, U.NY, U.NZ, U.NT
    end

	function Base.size(t::TemporaryField)
		return t.NX,t.NY,t.NZ,t.NT
	end

	function Base.similar(U::Gaugefield) 
		Uout = Gaugefield(U.NX, U.NY, U.NZ, U.NT, U.β)
		return Uout
	end

	function Base.similar(t::TemporaryField)
		tout = TemporaryField(t.NX, t.NY, t.NZ, t.NT)
		return tout
	end

	function substitute_U!(a::Gaugefield, b::Gaugefield)
		NX, NY, NZ, NT = size(a)

		@batch for it in 1:NT
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

		@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							a[μ][ix,iy,iz,it],
							b[μ][ix,iy,iz,it] = 
							b[μ][ix,iy,iz,it],
							a[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end

		return nothing 
	end

	function identity_gauges(NX, NY, NZ, NT, β; kind_of_gaction = "wilson")
		U = Gaugefield(NX, NY, NZ, NT, β, kind_of_gaction = kind_of_gaction)
		
		for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							U[μ][ix,iy,iz,it] = eye3
						end
					end
				end
			end
		end

        return U
    end

    function random_gauges(NX, NY, NZ, NT, β; kind_of_gaction = "wilson", rng = Xoshiro())
		U = Gaugefield(NX, NY, NZ, NT, β, kind_of_gaction = kind_of_gaction)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							link = SMatrix{3,3,ComplexF64}(rand(rng, ComplexF64, 3, 3))
							link = proj_onto_SU3(link)
							U[μ][ix,iy,iz,it] = link
						end
					end
				end
			end
		end

		Sg = calc_gauge_action(U)
		U.Sg = Sg
        return U
    end

	function clear_U!(U::Gaugefield)
		NX, NY, NZ, NT = size(U)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							@inbounds U[μ][ix,iy,iz,it] = @SMatrix zeros(ComplexF64, 3, 3)
						end
					end
				end
			end
		end

		return nothing
	end

	function normalize!(U::Gaugefield)
		NX, NY, NZ, NT = size(U)

 		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						for μ in 1:4
							link = U[μ][ix,iy,iz,it]
							U[μ][ix,iy,iz,it] = proj_onto_SU3(link)
						end
					end
				end
			end
        end

        return nothing
    end

	include("./wilsonloops.jl")

	function plaquette(U::Gaugefield, μ, ν, site::SiteCoords)
		Nμ = size(U)[μ]
		Nν = size(U)[ν]
		siteμ = move(site, μ, 1, Nμ)
		siteν = move(site, ν, 1, Nν)
		plaq = U[μ][site] * U[ν][siteμ] * U[μ][siteν]' * U[ν][site]'
		return plaq
	end

	function plaquette_trace_sum(U::Gaugefield)
		space = 8
		plaq = zeros(Float64, nthreads() * space)
		NX, NY, NZ, NT = size(U)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						site = SiteCoords(ix, iy, iz, it)
						for μ in 1:3
							for ν in μ+1:4
								plaq[threadid() * space] += 
								real(tr(plaquette(U, μ, ν, site)))
							end
						end  
					end	
				end
			end
		end

		return sum(plaq)
	end

	function rect_trace_sum(U::Gaugefield)
		space = 8
		rect = zeros(Float64, nthreads() * space)
		NX, NY, NZ, NT = size(U)

		@batch for it in 1:NT
			for iz in 1:NZ
				for iy in 1:NY
					for ix in 1:NX
						site = SiteCoords(ix, iy, iz, it)
						for μ in 1:3
							for ν in μ+1:4
								rect[threadid()*space] +=
									real(tr(wilsonloop_top_right(U, μ, ν, site, 1, 2))) +
									real(tr(wilsonloop_top_right(U, μ, ν, site, 2, 1)))
							end
						end 
					end
				end
			end
		end

		return sum(rect)
	end

	include("staples.jl")
	include("clovers.jl")
	include("fieldstrength.jl")
	include("actions.jl")
	include("liefields.jl")

end	