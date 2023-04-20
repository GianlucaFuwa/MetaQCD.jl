module Gaugefields
	using Random
	using LinearAlgebra
	using StaticArrays
	using Base.Threads
	using Polyester

	using ..Utils

	struct Site_coords{T}
		x::T
		y::T
		z::T
		t::T
	end 
	
	function Base.:+(s::Site_coords, t::NTuple{4})
		x,y,z,t = (s.x,s.y,s.z,s.t) .+ t
		return x,y,z,t
	end
	
	@inline function getcoords(s::Site_coords)
		return (s.x,s.y,s.z,s.t)
	end
	
	@inline function move(s::Site_coords, μ, steps, lim)
		x,y,z,t = getcoords(s)
		if μ == 1
			x = mod1(x+steps, lim)
		elseif μ == 2
			y = mod1(y+steps, lim)
		elseif μ == 3
			z = mod1(z+steps, lim)
		elseif μ == 4
			t = mod1(t+steps, lim)
		end
		return Site_coords(x,y,z,t)
	end

	function Base.setindex!(U::Array{SMatrix{3,3,ComplexF64,9},4}, v, s::Site_coords)
		x,y,z,t = getcoords(s)
		@inbounds U[x,y,z,t] = v
		return nothing
	end

	@inline function Base.getindex(U::Array{SMatrix{3,3,ComplexF64,9},4}, s::Site_coords)
		x,y,z,t = getcoords(s)
		@inbounds return U[x,y,z,t] 
	end

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

		function Gaugefield(NX::T, NY::T, NZ::T, NT::T, β::Float64; kind_of_gaction="Wilson") where {T<:Integer}
			U = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 0)
			for μ = 1:4
				Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef, NX, NY, NZ, NT)
				fill!(Uμ, SMatrix{3,3,ComplexF64}(zeros(ComplexF64,3,3)))
				push!(U, Uμ)
			end
			NV = NX*NY*NZ*NT

			type_of_gaction = kind_of_gaction
			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new(U, NX, NY, NZ, NT, NV, 3, β, type_of_gaction, Sg, CV)
		end
	end

	struct Temporary_field
		U::Vector{Array{SMatrix{3,3,ComplexF64,9},4}}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64

		function Temporary_field(NX, NY, NZ, NT)
			U = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 0)
			for μ = 1:4
				Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef, NX, NY, NZ, NT)
				fill!(Uμ, SMatrix{3,3,ComplexF64}(zeros(ComplexF64,3,3)))
				push!(U, Uμ)
			end
			NV = NX*NY*NZ*NT
			NC = 3
			return new(U, NX, NY, NZ, NT, NV, NC)
		end

		function Temporary_field(g::Gaugefield)
			NX, NY, NZ, NT = size(g)
			return Temporary_field(NX, NY, NZ, NT)
		end
	end

	function Base.setindex!(g::Gaugefield, v, μ)
        @inbounds g.U[μ] = v
		return nothing
    end

	@inline function Base.getindex(g::Gaugefield, μ)
        @inbounds return g.U[μ]
    end

	function Base.setindex!(t::Temporary_field, v, μ)
		@inbounds t.U[μ] = v
		return nothing
	end

	@inline function Base.getindex(t::Temporary_field, μ)
		@inbounds return t.U[μ]
	end

	function Base.getproperty(g::Gaugefield, p::Symbol)
		if p === :Sg 
			return getfield(g, :Sg)[]
		elseif p === :CV
			return getfield(g, :CV)[]
		else 
			return getfield(g, p)
		end
	end

	function Base.setproperty!(g::Gaugefield, p::Symbol, val)
		if p === :Sg 
			return getfield(g, :Sg)[] = val
		elseif p === :CV
			return getfield(g, :CV)[] = val
		else 
			return setfield!(g, p, val)
		end
	end

	function get_β(g::Gaugefield)
		return g.β
	end

	function get_gaction(g::Gaugefield)
		return g.kind_of_gaction
	end

	function get_Sg(g::Gaugefield)
		return g.Sg[]
	end

	function get_CV(g::Gaugefield)
		return g.CV[]
	end

	function set_Sg!(g::Gaugefield, val)
		g.Sg[] = val
		return nothing
	end

	function set_CV!(g::Gaugefield, val)
		g.CV[] = val
		return nothing
	end

	function add_Sg!(g::Gaugefield, val)
		g.Sg[] += val
		return nothing
	end

	function add_CV!(g::Gaugefield, val)
		g.CV[] += val
		return nothing
	end

	function Base.size(g::Gaugefield)
        return g.NX, g.NY, g.NZ, g.NT
    end

	function Base.size(t::Temporary_field)
		return t.NX,t.NY,t.NZ,t.NT
	end

	function Base.similar(g::Gaugefield) 
		gout = Gaugefield(g.NX, g.NY, g.NZ, g.NT, g.β)
		return gout
	end

	function Base.similar(t::Temporary_field)
		tout = Temporary_field(t.NX, t.NY, t.NZ, t.NT)
		return tout
	end

	function substitute_U!(a::T, b::T) where {T<:Gaugefield}
		NX, NY, NZ, NT = size(a)
		for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
						for μ = 1:4
							a[μ][ix,iy,iz,it] = b[μ][ix,iy,iz,it]
						end
					end
				end
			end
		end
		return nothing 
	end

	function Base.adjoint(U::Gaugefield)
		adj = similar(U)
		for μ = 1:4
			adj[μ] = U[μ]' 
        end
		return adj
	end

	function IdentityGauges(NX, NY, NZ, NT, β; gaction="Wilson")
		U = Gaugefield(NX, NY, NZ, NT, β, kind_of_gaction = gaction)
		
		for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
						for μ = 1:4
							U[μ][ix,iy,iz,it] = SMatrix{3,3}(I)
						end
					end
				end
			end
		end

        return U
    end

    function RandomGauges(NX, NY, NZ, NT, β; gaction="Wilson", rng::Xoshiro=Xoshiro())
		U = Gaugefield(NX, NY, NZ, NT, β, kind_of_gaction = gaction)

		for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
						for μ = 1:4
							link = SMatrix{3,3}( rand(rng,ComplexF64,3,3) .- 0.5 )
							link = proj_onto_SU3(link)
							U[μ][ix,iy,iz,it] = link
						end
					end
				end
			end
		end

		Sg = calc_GaugeAction(U)
		U.Sg = Sg

        return U
    end

	function clear_U!(U::Gaugefield)
		NX, NY, NZ, NT = size(U)
		for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
						for μ = 1:4
							@inbounds U[μ][ix,iy,iz,it] = zeros(3,3)
						end
					end
				end
			end
		end
		return nothing
	end

	function normalize!(U::Gaugefield)
		NX, NY, NZ, NT = size(U)
 		@batch for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
						for μ = 1:4
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

	function plaquette(U::T, μ, ν, site::Site_coords) where {T<:Gaugefield}
		Nμ = size(U)[μ]
		Nν = size(U)[ν]
		siteμ = move(site, μ, 1, Nμ)
		siteν = move(site, ν, 1, Nν)
		plaq = U[μ][site] * U[ν][siteμ] * U[μ][siteν]' * U[ν][site]'
		return plaq
	end

	function plaquette_tracedsum(U::T) where {T<:Gaugefield}
		space = 8
		plaq = zeros(Float64, nthreads()*space)
		NX, NY, NZ, NT = size(U)
		@batch for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
						site = Site_coords(ix,iy,iz,it)
						for μ = 1:3
							for ν = μ+1:4
								plaq[threadid()*space] += 
									real(tr( plaquette(U, μ, ν, site) ))
							end
						end  
					end	
				end
			end
		end
		return sum(plaq)
	end

	function rect_tracedsum(U::T) where {T<:Gaugefield}
		space = 8
		rect = zeros(Float64, nthreads()*space)
		NX, NY, NZ, NT = size(U)
		@batch for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
						site = Site_coords(ix,iy,iz,it)
						for μ = 1:3
							for ν = μ+1:4
								rect[threadid()*space] += 
									real(tr( wilsonloop_top_right(U, μ, ν, site, 1, 2) )) +
									real(tr( wilsonloop_top_right(U, μ, ν, site, 2, 1) ))
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

	include("actions.jl")

end
		
		