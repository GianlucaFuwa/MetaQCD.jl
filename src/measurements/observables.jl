module Observables 
    using LinearAlgebra
    using StaticArrays
	#using Base.Threads
	using Polyester
 
	import ..System_parameters: Params
	import ..Utils: exp_iQ
    import ..Gaugefields: Gaugefield

    l_vector = []
	vx = @SVector [1,0,0,0]
    push!(l_vector,vx)
    vy = @SVector [0,1,0,0]
    push!(l_vector,vy)
    vz = @SVector [0,0,1,0]
    push!(l_vector,vz)
	vt = @SVector [0,0,0,1]
    push!(l_vector,vt)

	function calc_Plaq(g::Gaugefield,μ::Int64,v::Int64,origin::SVector{4,Int64})
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

    function calc_PlaqSum(g::Gaugefield)
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