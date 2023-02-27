module Gaugefields
	using Random
	using LinearAlgebra
	using StaticArrays
	using Base.Threads
	using Polyester

	import ..System_parameters:Params
	import ..Utils:exp_iQ

	struct Site_coords{T} where T<:Int
        x::T
        y::T
        z::T
        t::T
    end 

	function move(s::Site_coords,dir,N,steps=1)
		x = mod1(s.x+(dir==1)*steps,N)
		return Site_coords(x,s.y,s.z,s.t)
	end
	
	mutable struct Gaugefield
		U::Array{Array{SMatrix{3,3,ComplexF64,9},4},1}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64
		β::Float64	
		CV::Float64
		
		function Gaugefield(p::Params)
			NX = p.L[1]
			NY = p.L[2]
			NZ = p.L[3]
			NT = p.L[4]
			NV = NX*NY*NZ*NT
			U = Array{Array{SMatrix{3,3,ComplexF64,9},4},1}(undef,0)
			β = p.β
			return new(U,NX,NY,NZ,NT,NV,3,β,0.0)
		end

		function Gaugefield(NX,NY,NZ,NT,β)
			U = zeros(ComplexF64,3,3,4,NX,NY,NZ,NT)
			NV = NX*NY*NZ*NT
			NC = 3
			return new(U,NX,NY,NZ,NT,NV,NC,β,0.0)
		end
	end

	function Base.setindex!(g::Gaugefield,v,μ)
        @inbounds g.U[μ] = v
		return
    end

	function Base.setindex!(g::Gaugefield,v,μ,ix,iy,iz,it) 
        @inbounds g.U[μ][ix,iy,iz,it] = v
		return
    end

	function Base.setindex!(g::Gaugefield,v,μ,ix,iy,iz,it,i1,i2) 
        @inbounds g.U[μ][ix,iy,iz,it][i1,i2] = v
		return 
    end

	function Base.setindex!(g::Gaugefield,v,μ,site::CartesianIndex{4}) 
        @inbounds g.U[μ][site] = v
		return 
    end

	@inline function Base.getindex(g::Gaugefield,μ)
        @inbounds return g.U[μ]
    end

	@inline function Base.getindex(g::Gaugefield,μ,ix,iy,iz,it)
        @inbounds return g.U[μ][ix,iy,iz,it]
    end

	@inline function Base.getindex(g::Gaugefield,μ,ix,iy,iz,it,i1,i2)
        @inbounds return g.U[μ][ix,iy,iz,it][i1,i2]
    end

	@inline function Base.getindex(g::Gaugefield,μ,site::CartesianIndex{4})
        @inbounds return g.U[μ][site]
    end

	function substitute_U!(a::T, b::T) where {T<:Gaugefield}
		NT = a.NT
		NZ = a.NZ
		NY = a.NY
		NX = a.NX
		for μ=1:4
			for it = 1:NT
			for iz=1:NZ
			for iy=1:NY
			for ix = 1:NX
			@inbounds a[μ][ix,iy,iz,it] = b[μ][ix,iy,iz,it]
			end
			end
			end
			end
		end
		return 
	end


	function Base.adjoint(g::Gaugefield)
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		adj = similar(g.U)
		for it=1:NT
		for iz=1:NZ
		for iy=1:NY
		for ix=1:NX
			for μ=1:4
				adj[μ][ix,iy,iz,it] = g.U[μ][ix,iy,iz,it]' 
			end
		end
		end
		end
        end
		return adj
	end

	function IdentityGauges!(g::Gaugefield)
		U = Array{Array{SMatrix{3,3,ComplexF64,9},4},1}(undef,0)
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		for μ=1:4
			Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef,NX,NY,NZ,NT)
			for it=1:NT
			for iz=1:NZ
			for iy=1:NY
			for ix=1:NX
				Uμ[ix,iy,iz,it] = SMatrix{3,3}(I)
			end
			end
			end
			end
		push!(U,Uμ)
		end
		g.U = U
        return nothing
    end

    
    function RandomGauges!(g::Gaugefield,rng::Xoshiro)
		U = Array{Array{SMatrix{3,3,ComplexF64,9},4},1}(undef,0)
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		for μ=1:4
			Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef,NX,NY,NZ,NT)
			for it=1:NT
			for iz=1:NZ
			for iy=1:NY
			for ix=1:NX
				col1 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
				col1 /= norm(col1)
				col2 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
				col2 -= (col1'*col2)*col1
				col2 /= norm(col2)
				col3 = cross(conj(col1),conj(col2))
				Uμ[ix,iy,iz,it] = [col1 col2 col3]
			end
			end
			end
			end
		push!(U,Uμ)
		end
		g.U = U
        return nothing
    end

	function normalize!(g::Gaugefield)
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
 		@batch for it=1:NT
		for iz=1:NZ
		for iy=1:NY
		for ix=1:NX
			for μ=1:4
				col1 = g[μ][ix,iy,iz,it][:,1]
				col2 = g[μ][ix,iy,iz,it][:,2]
				col3 = g[μ][ix,iy,iz,it][:,3]
				col1 /= norm(col1)
				col2 -= (col1'*col2)*col1
				col2 /= norm(col2)
				col3 -= (col1'*col3)*col1 + (col2'*col3)*col2
				col3 /= norm(col3)
				g[μ][ix,iy,iz,it] = [col1 col2 col3]
			end
		end
		end
		end
        end
        return nothing
    end

	function calc_Sgwils(g::Gaugefield)
		space = 8
		Sg = zeros(Float64,nthreads()*space)
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		@batch for it=1:NT
		for iz=1:NZ
		for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
		for ix=1:NX; ix_plu = mod1(ix+1,NX);
			Sg[threadid()*space] += 3.0 - real(tr(g[1][ix,iy,iz,it]*g[2][ix_plu,iy,iz,it]*g[1][ix,iy_plu,iz,it]'*g[2][ix,iy,iz,it]')) +
				3.0 - real(tr(g[2][ix,iy,iz,it]*g[3][ix,iy_plu,iz,it]*g[2][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]')) +
				3.0 - real(tr(g[3][ix,iy,iz,it]*g[1][ix,iy,iz_plu,it]*g[3][ix_plu,iy,iz,it]'*g[1][ix,iy,iz,it]')) +
				3.0 - real(tr(g[1][ix,iy,iz,it]*g[4][ix_plu,iy,iz,it]*g[1][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]')) +
				3.0 - real(tr(g[2][ix,iy,iz,it]*g[4][ix,iy_plu,iz,it]*g[2][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]')) +
				3.0 - real(tr(g[3][ix,iy,iz,it]*g[4][ix,iy,iz_plu,it]*g[3][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]'))
		end
		end
		end
		end
		return sum(Sg)/3/(NX*NY*NZ*NT)
	end
	
	l_vector = []
	vx = @SVector [1,0,0,0]
    push!(l_vector,vx)
    vy = @SVector [0,1,0,0]
    push!(l_vector,vy)
    vz = @SVector [0,0,1,0]
    push!(l_vector,vz)
	vt = @SVector [0,0,0,1]
    push!(l_vector,vt)

	function calc_plaq(g::Gaugefield,μ::Int64,v::Int64,origin::SVector{4,Int64})
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		μ_nb_p = mod1.(origin + l_vector[μ],(NX,NY,NZ,NT))
		v_nb_p = mod1.(origin + l_vector[v],(NX,NY,NZ,NT))
		plaq = g[μ][CartesianIndex(Tuple(origin))] *g[v][CartesianIndex(Tuple(μ_nb_p))] *
			   g[μ][CartesianIndex(Tuple(v_nb_p))]'*g[v][CartesianIndex(Tuple(origin))]'
		return plaq
	end

	function calc_plaq(g::Gaugefield)
		space = 8
		plaq = zeros(Float64,nthreads()*space)
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		@batch for it=1:NT
		for iz=1:NZ
		for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
		for ix=1:NX; ix_plu = mod1(ix+1,NX);
			plaq[threadid()*space] += tr(g[1][ix,iy,iz,it]*g[2][ix_plu,iy,iz,it]*g[1][ix,iy_plu,iz,it]'*g[2][ix,iy,iz,it]') +
				  tr(g[2][ix,iy,iz,it]*g[3][ix,iy_plu,iz,it]*g[2][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]') +
				  tr(g[3][ix,iy,iz,it]*g[1][ix,iy,iz_plu,it]*g[3][ix_plu,iy,iz,it]'*g[1][ix,iy,iz,it]') +
				  tr(g[1][ix,iy,iz,it]*g[4][ix_plu,iy,iz,it]*g[1][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +
				  tr(g[2][ix,iy,iz,it]*g[4][ix,iy_plu,iz,it]*g[2][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +
				  tr(g[3][ix,iy,iz,it]*g[4][ix,iy,iz_plu,it]*g[3][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]')
		end
		end
		end
		end
		return sum(plaq)
	end
	
	function calc_staplesum(g::Gaugefield,μ::Int64,origin::SVector{4,Int64})
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		μ_nb_p = mod1.(origin + l_vector[μ],(NX,NY,NZ,NT))
		staple = zeros(ComplexF64,3,3)
		for v=1:4
			if v == μ
				continue
			end
			v_nb_p = mod1.(origin + l_vector[v],(NX,NY,NZ,NT))
			v_nb_n = mod1.(origin - l_vector[v],(NX,NY,NZ,NT))
			vμ_nb_np = mod1.(origin - l_vector[v] + l_vector[μ],(NX,NY,NZ,NT))
			staple += g[v][CartesianIndex(Tuple(origin))]  * g[μ][CartesianIndex(Tuple(v_nb_p))]' * g[v][CartesianIndex(Tuple(μ_nb_p))]' +
					  g[v][CartesianIndex(Tuple(v_nb_n))]' * g[μ][CartesianIndex(Tuple(v_nb_n))]' * g[v][CartesianIndex(Tuple(vμ_nb_np))]
		end
		return staple
	end

	function calc_cloversum(g::Gaugefield,μ::Int64,v::Int64,origin::SVector{4,Int64})
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		μ_nb_p = mod1.(origin + l_vector[μ],(NX,NY,NZ,NT))
		μ_nb_n = mod1.(origin - l_vector[μ],(NX,NY,NZ,NT))
		v_nb_p = mod1.(origin + l_vector[v],(NX,NY,NZ,NT))
		v_nb_n = mod1.(origin - l_vector[v],(NX,NY,NZ,NT))
		vμ_nb_np = mod1.(origin - l_vector[v] + l_vector[μ],(NX,NY,NZ,NT))
		vμ_nb_pn = mod1.(origin + l_vector[v] - l_vector[μ],(NX,NY,NZ,NT))
		vμ_nb_nn = mod1.(origin - l_vector[v] - l_vector[μ],(NX,NY,NZ,NT))

		clover = g[μ][CartesianIndex(Tuple(origin))] *g[v][CartesianIndex(Tuple(μ_nb_p))] *  g[μ][CartesianIndex(Tuple(v_nb_p))]' *g[v][CartesianIndex(Tuple(origin))]' -
				 g[μ][CartesianIndex(Tuple(origin))] *g[v][CartesianIndex(Tuple(vμ_nb_np))]'*g[μ][CartesianIndex(Tuple(v_nb_n))]' *g[v][CartesianIndex(Tuple(v_nb_n))] - 
				 g[μ][CartesianIndex(Tuple(μ_nb_n))]'*g[v][CartesianIndex(Tuple(vμ_nb_nn))]'*g[μ][CartesianIndex(Tuple(vμ_nb_nn))]*g[v][CartesianIndex(Tuple(v_nb_n))] +
				 g[μ][CartesianIndex(Tuple(μ_nb_n))]'*g[v][CartesianIndex(Tuple(μ_nb_n))] *  g[μ][CartesianIndex(Tuple(vμ_nb_pn))]*g[v][CartesianIndex(Tuple(origin))]'
		return clover
	end

	function calc_cloverAH(g::Gaugefield,μ::Int64,v::Int64,origin::SVector{4,Int64})
		clover = calc_cloversum(g,μ,v,origin)
		return -im*( (clover-clover') - 1/3*tr(clover-clover')*I )
	end

	function top_charge_density(g::Gaugefield)
		NX = g.NX
		NY = g.NY
		NZ = g.NZ
		NT = g.NT
		q = zeros(ComplexF64,NX,NY,NZ,NT)

		@threads for it=1:NT
		for iz=1:NZ
		for iy=1:NY
		for ix=1:NX
			C01 = calc_cloverAH(g,4,1,SA[ix,iy,it,iz])
			C23 = calc_cloverAH(g,2,3,SA[ix,iy,it,iz])
			C02 = calc_cloverAH(g,4,2,SA[ix,iy,it,iz])
			C13 = calc_cloverAH(g,1,3,SA[ix,iy,it,iz])
			C03 = calc_cloverAH(g,4,3,SA[ix,iy,it,iz])
			C12 = calc_cloverAH(g,1,2,SA[ix,iy,it,iz])
			q[ix,iy,iz,it] += -1/256pi^2*tr( C01*C23 + C02*C13 + C03*C12 ) 
		end
		end
		end
		end
		return q
	end

	function top_charge(g::Gaugefield)
		q = top_charge_density(g)
		return real(sum(q))
	end

	function recalc_CV!(g::Gaugefield)
		g.CV = top_charge(g)
		return nothing
	end

	function substitute_U!(a::Gaugefield,b::Gaugefield)
        for μ=1:4
            copy!(a.U[μ][:,:,:,:],b.U[μ])
        end
    end

end
		
		