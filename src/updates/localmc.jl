struct Local_update <: AbstractUpdate
	ϵ::Base.RefValue{Float64}
	multi_hit::Int64
	metro_target_acc::Float64
	metro_norm::Float64
	meta_enabled::Bool
end

get_ϵ(loc::Local_update) = loc.ϵ[]
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
		numaccepts = local_metro_sweep_meta!(U, Bias, rng, ϵ, updatemethod.multi_hit)
	else
		numaccepts = local_metro_sweep!(U, rng, ϵ, updatemethod.multi_hit)
	end
	adjust_ϵ!(updatemethod, numaccepts)
	return numaccepts / U.NV / 4.0
end

function local_metro!(U::Gaugefield, μ::Int64, origin::Site_coords, rng::Xoshiro, ϵ::Float64, multi_hit)
	accept = 0
	for hit = 1:multi_hit
		X = gen_SU3_matrix(rng, ϵ)
		Sg = get_Sg(U)
		ΔSg = local_action_diff(U, μ, origin, X)
		if rand(rng) ≤ exp(-ΔSg)
			#println_verbose3(verbose,"Accepted")
			U.Sg = Sg + ΔSg
			U[μ][origin] = X * U[μ][origin]
			accept += 1
		else
			#println_verbose3(verbose,"Rejected")
		end
	end
	return accept
end

function local_metro_meta!(U::Gaugefield, bias::Bias_potential, μ::Int64, origin::Site_coords, rng::Xoshiro, ϵ::Float64, multi_hit)
	accept = 0
	for hit = 1:multi_hit
		X = gen_SU3_matrix(rng, ϵ)
		Sg = get_Sg(U)
		CV = get_CV(U)
		ΔCV = local_cv_diff(U, μ, origin, X)
		println_verbose3(verbose, "ΔCV = $ΔCV")
		
		ΔSg = local_action_diff(U, μ, origin, X)
		println_verbose3(verbose, "ΔSg = $ΔSg")
		ΔV = DeltaV(bias, CV, CV + ΔCV)
		println_verbose3(verbose, "ΔV = $ΔV")
		if rand(rng) ≤ exp(-ΔSg-ΔV) 
			U[μ][origin] = X * U[μ][origin]
			U.Sg = Sg + ΔSg
			U.CV = CV + ΔCV 
			println_verbose3(verbose, "Accepted")
			if ~bias.is_static
				update_bias!(bias, CV + ΔCV)
			end
			accept += 1
		else
			#println_verbose3(verbose,"Rejected")
		end
	end
	return accept
end

function local_metro_sweep!(U::Gaugefield, rng::Xoshiro, ϵ::Float64, multi_hit::Int64)
	NX, NY, NZ, NT = size(U)
	numaccepts = 0
	for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
					for μ = 1:4
						accept = local_metro!(U, μ, Site_coords(ix,iy,iz,it), rng, ϵ, multi_hit)
						numaccepts += accept
					end
				end
			end
		end
	end
	return numaccepts
end

function local_metro_sweep_meta!(U::Gaugefield, bias::Bias_potential, rng::Xoshiro, ϵ::Float64, multi_hit::Int64)
	NX, NY, NZ, NT = size(U)
	numaccepts = 0
	for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
					for μ = 1:4
						accept = local_metro_meta!(U, bias, μ, Site_coords(ix,iy,iz,it), rng, ϵ, multi_hit)
						numaccepts += accept
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

function adjust_ϵ!(loc::Local_update, numaccepts)
	loc.ϵ[] += (numaccepts*loc.metro_norm - loc.metro_target_acc) * 0.1
	return nothing
end
