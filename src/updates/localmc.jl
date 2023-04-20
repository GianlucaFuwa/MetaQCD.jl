struct Local_update <: AbstractUpdate
	ϵ::Float64
	meta_enabled::Bool
end

get_ϵ(loc::Local_update) = loc.ϵ
is_meta(loc::Local_update) = loc.meta_enabled

function update!(
	updatemethod::T,
	U::Gaugefield,
	rng::Xoshiro,
	verbose::Verbose_level;
	Bias::Union{Nothing,Bias_potential}=nothing,
	metro_test::Bool = true,
	) where {T<:Local_update}
	ϵ = get_ϵ(updatemethod)
	if is_meta(updatemethod)
		numaccepts = local_metro_sweep_meta!(U, Bias, rng, ϵ)
	else
		numaccepts = local_metro_sweep!(U, rng, ϵ)
	end
	return numaccepts / U.NV / 4.0
end

function local_metro!(gfield::Gaugefield, μ::Int64, origin::Site_coords, rng::Xoshiro, ϵ::Float64)
	X = gen_SU3_matrix(rng, ϵ)
	Sg = get_Sg(gfield)
	ΔSg = local_action_diff(gfield, μ, origin, X)
	accept = rand(rng) ≤ exp(-ΔSg)
	if accept  
		#println_verbose3(verbose,"Accepted")
		gfield.Sg = Sg + ΔSg
		gfield[μ][origin] = X * gfield[μ][origin]
	else
		#println_verbose3(verbose,"Rejected")
	end
	return accept
end

function local_metro_meta!(gfield::Gaugefield, bias::Bias_potential, μ::Int64, origin::Site_coords, rng::Xoshiro, ϵ::Float64)
	X = gen_SU3_matrix(rng, ϵ)
	Sg = get_Sg(gfield)
	CV = get_CV(gfield)
	ΔCV = local_cv_diff(gfield, μ, origin, X)
	println_verbose3(verbose, "ΔCV = $ΔCV")
	
	ΔSg = local_action_diff(gfield, μ, origin, X)
	println_verbose3(verbose, "ΔSg = $ΔSg")
	ΔV = DeltaV(bias, CV, CV + ΔCV)
	println_verbose3(verbose, "ΔV = $ΔV")
	accept = rand(rng) ≤ exp(-ΔSg-ΔV) 
	if accept 
		gfield[μ][origin] = X * gfield[μ][origin]
		gfield.Sg = Sg + ΔSg
		gfield.CV = CV + ΔCV 
		println_verbose3(verbose, "Accepted")
		if ~b.is_static
			update_bias!(bias, CV + ΔCV)
		end
	else
		#println_verbose3(verbose,"Rejected")
	end
	return accept
end

function local_metro_sweep!(gfield::Gaugefield, rng::Xoshiro, ϵ::Float64)
	numaccepts = 0
	for it = 1:gfield.NT
		for iz = 1:gfield.NZ
			for iy = 1:gfield.NY
				for ix = 1:gfield.NX
					for μ = 1:4
						accept = local_metro!(gfield, μ, Site_coords(ix,iy,iz,it), rng, ϵ)
						numaccepts += ifelse(accept, 1, 0)
					end
				end
			end
		end
	end
	return numaccepts
end

function local_metro_sweep_meta!(gfield::Gaugefield, bias::Bias_potential, rng::Xoshiro, ϵ::Float64)
	numaccepts = 0
	for it = 1:gfield.NT
		for iz = 1:gfield.NZ
			for iy = 1:gfield.NY
				for ix = 1:gfield.NX
					for μ = 1:4
						accept = local_metro_meta!(gfield, bias, μ, Site_coords(ix,iy,iz,it), rng, ϵ)
						numaccepts += ifelse(accept, 1, 0)
					end
				end
			end
		end
	end
	return numaccepts
end

function local_action_diff(U::Gaugefield, μ::Int64, origin::Site_coords, X)
	link_old = U[μ][origin]
	A = staple(U, μ, origin)
	return -U.β/3 * real( tr((X * link_old - link_old) * A') ) 
end

function local_q_diff(U::Gaugefield, μ::Int64, origin::Site_coords, X)
	link_old = U[μ][origin]
	A = staple(U, μ, origin)
	return 1/2pi * imag( tr((X * link_old - link_old) * A') ) 
end
