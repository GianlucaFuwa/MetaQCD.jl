module LocalMC
	using Random
	using Printf
	using StaticArrays
	using LinearAlgebra
	
	import ..System_parameters:Params
	import ..Gaugefields:Gaugefield,calc_staplesum,substitute_U!
	import ..Metadynamics:Bias_potential,update_bias!,DeltaV
	import ..Utils:gen_proposal
	import ..Verbose_print:Verbose_,println_verbose

	function loc_metro!(gfield::Gaugefield,μ::Int64,origin::SVector{4, Int64},rng::Xoshiro,ϵ::Float64)
		X = gen_proposal(rng,ϵ)
		ΔSg = loc_action_diff(gfield,μ,origin,X)
		accept = rand(rng) ≤ exp(-ΔSg)
		if accept  
			gfield[μ][CartesianIndex(Tuple(origin))] = X*gfield[μ][CartesianIndex(Tuple(origin))]
		end
		return accept
	end
	
	function loc_metro_meta!(gfield::Gaugefield,bias::Bias_potential,μ::Int64,origin::SVector{4, Int64},rng::Xoshiro,ϵ::Float64)
		X = gen_proposal(rng,ϵ)
		CV = gfield.CV
		ΔCV = local_cv_diff(gfield,μ,origin,X)
		
		ΔSg = local_action_diff(gfield,μ,origin,X)
		ΔV = DeltaV(bias,CV,CV+ΔCV)
		accept = rand(rng) ≤ exp(-ΔSg-ΔV) 
		if accept 
			gfield[μ][CartesianIndex(Tuple(origin))] = X*gfield[μ][CartesianIndex(Tuple(origin))]
			gfield.Sg += ΔSg
			gfield.CV += ΔCV 
			if ~b.is_static
				update_bias!(bias,gfield.CV)
			end
		end
		return accept
	end

	function loc_metro_sweep!(gfield::Gaugefield,rng::Xoshiro,ϵ::Float64)
		numaccepts = 0
		for it=1:gfield.NT
		for iz=1:gfield.NZ
		for iy=1:gfield.NY
		for ix=1:gfield.NX
		for μ=1:4
			accept = loc_metro!(gfield,μ,SA[ix,iy,iz,it],rng,ϵ)
			numaccepts += ifelse(accept,1,0)
		end
		end
		end
		end
		end
		return numaccepts
	end

	function loc_metro_sweep_meta!(gfield::Gaugefield,bias::Bias_potential,rng::Xoshiro,ϵ::Float64)
		numaccepts = 0
		for it=1:gfield.NT
		for iz=1:gfield.NZ
		for iy=1:gfield.NY
		for ix=1:gfield.NX
		for μ=1:4
			accept = loc_metro_meta!(gfield,bias,μ,SA[ix,iy,iz,it],rng,ϵ)
			numaccepts += ifelse(accept,1,0)
		end
		end
		end
		end
		end
		return numaccepts
	end

	function PT_swap!(field0::Gaugefield,field1::Gaugefield,bias::Bias_potential,rng::Xoshiro)
		cv0 = field0.CV
		cv1 = field1.CV
		ΔV = DeltaV(bias,cv1,cv0)
		accept_swap = rand(rng) ≤ exp(ΔV)
		if accept_swap
			substitute_U!(field0,field1)
			field1.CV = cv0
			field0.CV = cv1
			if ~b.is_static
				update_bias!(bias,field1.CV)
			end
		end
		return accept_swap
	end 

	function local_action_diff(x::Gaugefield,μ::Int64,origin::SVector{4, Int64},X::Union{Matrix{ComplexF64},Adjoint{ComplexF64, Matrix{ComplexF64}}})
		link_old = x[μ][CartesianIndex(Tuple(origin))]
		A = calc_staplesum(x,μ,origin)
		return -x.β/3*real( tr((X*link_old - link_old)*A') ) 
	end
	
	function local_q_diff(x::Gaugefield,μ::Int64,origin::SVector{4, Int64},X::Union{Matrix{ComplexF64},Adjoint{ComplexF64, Matrix{ComplexF64}}})
		link_old = x[μ][CartesianIndex(Tuple(origin))]
		A = calc_staplesum(x,μ,origin)
		return 1/2pi*imag( tr((X*link_old - link_old)*A') ) 
	end

end