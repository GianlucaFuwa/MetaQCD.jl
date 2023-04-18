module Gaugefields
	using Random
	using LinearAlgebra
	using StaticArrays
	using Base.Threads
	using Polyester

	import ..System_parameters: Params
	import ..Utils: exp_iQ

	struct Site_coords{T}
		x::T
		y::T
		z::T
		t::T
	end 
	
	function Base.:+(s::Site_coords,t::NTuple{4})
		x,y,z,t = (s.x,s.y,s.z,s.t) .+ t
		return x,y,z,t
	end
	
	@inline function getcoords(s::Site_coords)
		return (s.x,s.y,s.z,s.t)
	end
	
	@inline function move(s::Site_coords,μ::Int64,steps::Int64,lim::Int64)
		x,y,z,t = getcoords(s)
		if μ == 1
			x = mod1(x+steps,lim)
		elseif μ == 2
			y = mod1(y+steps,lim)
		elseif μ == 3
			z = mod1(z+steps,lim)
		elseif μ == 4
			t = mod1(t+steps,lim)
		end
		return Site_coords(x,y,z,t)
	end

	function Base.setindex!(U::Array{SMatrix{3,3,ComplexF64,9},4},v,s::Site_coords)
		x,y,z,t = getcoords(s)
		@inbounds U[x,y,z,t] = v
		return nothing
	end

	@inline function Base.getindex(U::Array{SMatrix{3,3,ComplexF64,9},4},s::Site_coords)
		x,y,z,t = getcoords(s)
		@inbounds return U[x,y,z,t] 
	end

    struct Gaugefield_μ
        U::Array{SMatrix{3,3,ComplexF64,9},4}
        NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
        Ushifted::Array{SMatrix{3,3,ComplexF64,9},4}

        function Gaugefield_μ(NX::I,NY::I,NZ::I,NT::I) where {I<:Integer}
            NV = NX*NY*NZ*NT
            U = Array{SMatrix{3,3,ComplexF64,9},4}(undef,NX,NY,NZ,NT)
            fill!(U,SMatrix{3,3}(zeros(ComplexF64,3,3)))
            Ushifted = zero(U)
            return new(U,NX,NY,NZ,NT,NV,Ushifted)
        end
    end

    function Base.setindex!(g::Gaugefield_μ,v,i1,i2,i3,i4)
        @inbounds g.U[i1,i2,i3,i4] = v
		return nothing
    end

	@inline function Base.getindex(g::Gaugefield_μ,i1,i2,i3,i4)
        @inbounds return g.U[i1,i2,i3,i4]
    end

	struct Gaugefield
		U::Vector{Gaugefield_μ}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64

		β::Float64
		type_of_gaction::String
		Sg::Base.RefValue{Float64}
		CV::Base.RefValue{Float64}
		
		function Gaugefield(p::Params)
			NX = p.L[1]
			NY = p.L[2]
			NZ = p.L[3]
			NT = p.L[4]
			NV = NX*NY*NZ*NT
			Uμ = Gaugefield_μ(NX,NY,NZ,NT)
			U = Vector{Gaugefield_μ}(undef,0)
			for μ = 1:4
				push!(U,Uμ)
			end

			β = p.β
			type_of_gaction = p.type_of_gaction
			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new(U,NX,NY,NZ,NT,NV,3,β,type_of_gaction,Sg,CV)
		end

		function Gaugefield(NX::T,NY::T,NZ::T,NT::T,β::Float64) where {T<:Integer}
			Uμ = Gaugefield_μ(NX,NY,NZ,NT)
			U = Vector{Gaugefield_μ}(undef,0)
			for μ = 1:4
				push!(U,Uμ)
			end
			NV = NX*NY*NZ*NT

			type_of_gaction = "Wilson"
			Sg = Base.RefValue{Float64}(0.0)
			CV = Base.RefValue{Float64}(0.0)
			return new(U,NX,NY,NZ,NT,NV,3,β,type_of_gaction,Sg,CV)
		end
	end

	function Base.setindex!(g::Gaugefield,v,μ)
        @inbounds g.U[μ] = v
		return nothing
    end

	@inline function Base.getindex(g::Gaugefield,μ)
        @inbounds return g.U[μ]
    end

	function Base.getproperty(g::Gaugefield,p::Symbol)
		if p === :Sg 
			return getfield(g, :Sg)[]
		elseif p === :CV
			return getfield(g, :CV)[]
		else 
			return getfield(g, p)
		end
	end

	function Base.setproperty!(g::Gaugefield,p::Symbol,val)
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

	function get_Sg(g::Gaugefield)
		return g.Sg
	end

	function get_CV(g::Gaugefield)
		return g.CV
	end

	function set_Sg!(g::Gaugefield,val)
		g.Sg = val
		return nothing
	end

	function set_CV!(g::Gaugefield,val)
		g.CV = val
		return nothing
	end

	function add_Sg!(g::Gaugefield,val)
		g.Sg += val
		return nothing
	end

	function add_CV!(g::Gaugefield,val)
		g.CV += val
		return nothing
	end

	function Base.size(g::Union{Gaugefield_μ,Gaugefield})
        return g.NX,g.NY,g.NZ,g.NT
    end

    function Base.similar(g::Gaugefield_μ)
        gout = Gaugefield_μ(g.NX,g.NY,g.NZ,g.NT)
        return gout
    end

	function Base.similar(g::Gaugefield) 
		gout = Gaugefield(g.NX,g.NY,g.NZ,g.NT,g.β)
		return gout
	end

	function substitute_U!(a::T, b::T) where {T<:Union{Gaugefield_μ,Gaugefield}}
		for i=1:length(a.U)
			a.U[i] = b.U[i]
		end
		return nothing 
	end

	function shifted_U!(U::Gaugefield_μ, shift) 
		NX,NY,NZ,NT = size(U)
		for it = 1:NT
			it_shifted = it + shift[4]
			it_shifted += ifelse(it_shifted > NT, -NT, 0)
			it_shifted += ifelse(it_shifted < 1, NT, 0)
	
			for iz = 1:NZ
				iz_shifted = iz + shift[3]
				iz_shifted += ifelse(iz_shifted > NZ, -NZ, 0)
				iz_shifted += ifelse(iz_shifted < 1, NZ, 0)
				for iy = 1:NY
					iy_shifted = iy + shift[2]
					iy_shifted += ifelse(iy_shifted > NY, -NY, 0)
					iy_shifted += ifelse(iy_shifted < 1, NY, 0)
					for ix = 1:NX
						ix_shifted = ix + shift[1]
						ix_shifted += ifelse(ix_shifted > NX, -NX, 0)
						ix_shifted += ifelse(ix_shifted < 1, NX, 0)
						for k2 = 1:NC
							for k1 = 1:NC
								U.Ushifted[k1, k2, ix, iy, iz, it] =
									U[k1, k2, ix_shifted, iy_shifted, iz_shifted, it_shifted]
							end
						end
					end
				end
			end
			return nothing
		end

	function Base.adjoint(g::Gaugefield_μ)
		NX,NY,NZ,NT = size(g)
		adj = similar(g)
		for it=1:NT
			for iz=1:NZ
				for iy=1:NY
					for ix=1:NX
							adj[ix,iy,iz,it] = g[ix,iy,iz,it]' 
					end
				end
			end
        end
		return adj
	end

    function make_identity!(g::Gaugefield_μ)
        fill!(g.U,SMatrix{3,3,ComplexF64}(I))
    end

	function IdentityGauges(NX,NY,NZ,NT,β)
		g = Gaugefield(NX,NY,NZ,NT,β)
		
		for it=1:NT
			for iz=1:NZ
				for iy=1:NY
					for ix=1:NX
						for μ=1:4
							g[μ][ix,iy,iz,it] = SMatrix{3,3}(I)
						end
					end
				end
			end
		end

        return g
    end

    function RandomGauges(NX,NY,NZ,NT,β,rng::Xoshiro=Xoshiro())
		g = Gaugefield(NX,NY,NZ,NT,β)

		for it=1:NT
			for iz=1:NZ
				for iy=1:NY
					for ix=1:NX
						for μ=1:4
							col1 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
							col1 /= norm(col1)
							col2 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
							col2 -= (col1'*col2)*col1
							col2 /= norm(col2)
							col3 = cross(conj(col1),conj(col2))
							col3 /= norm(col3)
							g[μ][ix,iy,iz,it] = [col1 col2 col3]
						end
					end
				end
			end
		end
		g.Sg = calc_GaugeAction(g)
        return g
    end

	function clear_U!(g::Gaugefield_μ)
		fill!(g.U,SMatrix{3,3}( zeros(ComplexF64,3,3) )) 
		return nothing
	end

	function normalize!(g::Gaugefield_μ)
		NX,NY,NZ,NT = size(g)
 		@batch per=thread for it=1:NT
			for iz=1:NZ
				for iy=1:NY
					for ix=1:NX
						col1 = g[ix,iy,iz,it][:,1]
						col2 = g[ix,iy,iz,it][:,2]
						col3 = g[ix,iy,iz,it][:,3]
						col1 /= norm(col1)
						col2 -= (col1'*col2)*col1
						col2 /= norm(col2)
						col3 -= (col1'*col3)*col1 + (col2'*col3)*col2
						col3 /= norm(col3)
						g[ix,iy,iz,it] = [col1 col2 col3]
					end
				end
			end
        end
        return nothing
    end

    function LinearAlgebra.tr(g::Gaugefield_μ)
        NX,NY,NZ,NT = size(g)
        s = 0.0
        for it=1:NT
			for iz=1:NZ
				for iy=1:NY
					for ix=1:NX
						@simd for k = 1:3
                            s += g[ix,iy,iz,it][k,k]
					end
				end
			end
		end
        return s
    end

    function LinearAlgebra.mul!(c::T,a::T,b::T) where {T<:Gaugefield_μ}
        NX,NY,NZ,NT = size(c)
        for it=1:NT
			for iz=1:NZ
				for iy=1:NY
					for ix=1:NX
                        c[ix,iy,iz,it] = a[ix,iy,iz,it] * b[ix,iy,iz,it]
					end
				end
			end
		end
        return nothing
    end

	function plaquette(g::T,μ,ν,site::Site_coords) where {T<:Gaugefield}
		Nμ = size(g)[μ]
		Nν = size(g)[ν]
		siteμ = move(site,μ,1,Nμ)
		siteν = move(site,ν,1,Nν)
		plaq = g[μ][site] * g[ν][siteμ] * g[μ][siteν]' * g[ν][site]'
		return plaq
	end

    function plaquette_trsum(g::T,temp::T) where {T<:Gaugefield,T1<:Gaugefield_μ}
		plaq = 0.0
        for μ = 1:3
            for ν = μ:4
                substitute_U!(temp,g[μ])
                shifted_U!(g[μ])
                mul!(temp,temp,g[ν])
                mul!(temp,temp,g[2])
                mul!(temp,temp,g[3])
                mul!(temp,temp,g[4])
        plaq += tr(temp)
		return real(plaq*0.5)
	end

	function rect_trsum(g::T) where {T<:Gaugefield}
		space = 8
		rect = zeros(ComplexF64,nthreads()*space)
		NX,NY,NZ,NT = size(g)
		@batch for it=1:NT
			for iz=1:NZ
				for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
					for ix=1:NX; ix_plu = mod1(ix+1,NX);
					plaq[threadid()*space]+=tr(g[1][ix,iy,iz,it]*g[2][ix_plu,iy,iz,it]*g[1][ix,iy_plu,iz,it]'*g[2][ix,iy,iz,it]') +
											tr(g[1][ix,iy,iz,it]*g[3][ix_plu,iy,iz,it]*g[1][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]') +
											tr(g[2][ix,iy,iz,it]*g[3][ix,iy_plu,iz,it]*g[2][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]') +
											tr(g[1][ix,iy,iz,it]*g[4][ix_plu,iy,iz,it]*g[1][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +	
											tr(g[2][ix,iy,iz,it]*g[4][ix,iy_plu,iz,it]*g[2][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +	
											tr(g[3][ix,iy,iz,it]*g[4][ix,iy,iz_plu,it]*g[3][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') 
					end
				end
			end
		end
		return real(sum(plaq))
	end

	function staple_plaq(g::T,μ,site::Site_coords) where {T<:Gaugefield}
		Nμ = size(g)[μ]
		Nν = size(g)[ν]
		siteμp = move(site,μ,1,Nμ)
		staple = SMatrix{3,3,ComplexF64,9}(I)
		for ν=1:4
			if ν == μ
				continue
			end
			siteνp = move(site,ν,1,Nν)
			siteνn = move(site,ν,-1,Nν)
			siteμpνn = move(siteμp,ν,-1,Nν)
			staple *= g[ν][site]  * g[μ][siteνp]' * g[ν][siteμp]' +
					  g[ν][siteνn]' * g[μ][siteνn]' * g[ν][siteμpνn]
		end
		return staple
	end

	function recalc_GaugeAction!(g::T) where {T<:Gaugefield}
		g.Sg = calc_GaugeAction(g)
		return nothing
	end

	function calc_GaugeAction(g::T) where {T<:Gaugefield}
        if g.type_of_gaction == "plaq"
            Sg = Sg_wils_plaq(g)
            Sg *= g.β
        else
            error("type_of_gaction $(g.type_of_gaction) is not supported!")
        end
        return Sg
    end

    function Sg_wils_plaq(g::T) where {T<:Gaugefield}
		P = plaquette_trsum(g)
		return g.β * (6*g.NV - 1/3*P)
	end

	function Sg_wils_improved(g::T) where {T<:Gaugefield}
		P = plaquette_trsum(g)
		R = rect_trsum(g)
		u0sq = sqrt(1/3*P)
		Sg_plaq = 6*g.NV - 1/3*P
		Sg_rect = 24*g.NV - 1/3*R
		Sg_improved = g.β*( 5/3*Sg_plaq - 1/(12u0sq)*Sg_rect ) 
		return Sg_improved
	end

	function polyakov_tr(g::Gaugefield)
		NX,NY,NZ,NT = size(g)
		poly = 0.0 + 0.0im
		@batch for it=1:NT
			for iz=1:NZ
				for iy=1:NY
					for ix=1:NX
						polymat = g[4][ix,iy,iz,it]
						for t=1:NT-1
							polymat *= g[4][ix,iy,iz,mod1(it+t,NT)]
						end
						poly += tr(polymat)
					end
				end
			end
		end
		return tr(poly)
	end 

end
		
		