struct MetroUpdate <: AbstractUpdate
	ϵ::Base.RefValue{Float64}
	multi_hit::Int64
	metro_target_acc::Float64
	metro_norm::Float64
	meta_enabled::Bool

	function MetroUpdate(U::Gaugefield, ϵ, multi_hit, metro_target_acc, meta_enabled)
		metro_norm = 1 / (U.NV * 4 * multi_hit)
		return new(
			Base.RefValue{Float64}(ϵ), multi_hit, metro_target_acc, metro_norm,
			meta_enabled
		)
	end
end

function update!(
	updatemethod::MetroUpdate,
	U::Gaugefield,
	rng,
	verbose::VerboseLevel;
	Bias = nothing,
	metro_test = true,
)
	ϵ = updatemethod.ϵ[]

	if updatemethod.meta_enabled
		numaccepts = metro_sweep_meta!(U, Bias, rng, ϵ, updatemethod.multi_hit)
	else
		numaccepts = metro_sweep!(U, rng, ϵ, updatemethod.multi_hit)
	end

	adjust_ϵ!(updatemethod, numaccepts)
	return numaccepts * updatemethod.metro_norm
end

function metro!(
	U::Gaugefield,
	μ,
	origin::SiteCoords,
	rng,
	ϵ,
	multi_hit,
)
	accept = 0

	for hit in 1:multi_hit
		X = gen_SU3_matrix(rng, ϵ)
		ΔSg = local_action_diff(U, μ, origin, X)

		if rand(rng) ≤ exp(-ΔSg)
			#println_verbose3(verbose,"Accepted")
			U.Sg += ΔSg
			U[μ][origin] = X * U[μ][origin]
			accept += 1
		else
			#println_verbose3(verbose,"Rejected")
		end

	end

	return accept
end

function metro_meta!(
	U::Gaugefield,
	bias::BiasPotential,
	μ,
	origin::SiteCoords,
	rng,
	ϵ,
	multi_hit,
)
	accept = 0

	for hit in 1:multi_hit
		X = gen_SU3_matrix(rng, ϵ)
		Sg = get_Sg(U)
		CV = U.CV
		ΔCV = local_cv_diff(U, μ, origin, X)
		println_verbose3(verbose, "ΔCV = $ΔCV")
		
		ΔSg = local_action_diff(U, μ, origin, X)
		println_verbose3(verbose, "ΔSg = $ΔSg")
		ΔV = DeltaV(bias, CV, CV + ΔCV)

		println_verbose3(verbose, "ΔV = $ΔV")

		if rand(rng) ≤ exp(-ΔSg-ΔV) 
			U[μ][origin] = X * U[μ][origin]
			U.Sg += ΔSg
			U.CV += ΔCV 
			println_verbose3(verbose, "Accepted")

			if !bias.is_static
				update_bias!(bias, CV + ΔCV)
			end

			accept += 1
		else
			#println_verbose3(verbose,"Rejected")
		end

	end

	return accept
end

function metro_sweep!(
	U::Gaugefield,
	rng,
	ϵ,
	multi_hit,
)
	NX, NY, NZ, NT = size(U)
	numaccepts = 0

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)
					for μ in 1:4
						accept = metro!(U, μ, site, rng, ϵ, multi_hit)
						numaccepts += accept
					end
				end
			end
		end
	end

	return numaccepts
end

function metro_sweep_meta!(
	U::Gaugefield,
	bias::BiasPotential,
	rng,
	ϵ,
	multi_hit,
)
	NX, NY, NZ, NT = size(U)
	numaccepts = 0

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)
					for μ in 1:4
						accept = metro_meta!(U, bias, μ, site, rng, ϵ, multi_hit)
						numaccepts += accept
					end
				end
			end
		end
	end

	return numaccepts
end

function local_action_diff(U::Gaugefield, μ, origin::SiteCoords, X)
	link_old = U[μ][origin]
	A = staple(U, μ, origin)
	return -U.β/3 * real(tr((X * link_old - link_old) * A')) 
end

function local_q_diff(U::Gaugefield, μ, origin::SiteCoords, X)
	link_old = U[μ][origin]
	A = staple(U, μ, origin)
	return 1/2pi * imag(tr((X * link_old - link_old) * A')) 
end

function adjust_ϵ!(updatemethod::MetroUpdate, numaccepts)
	updatemethod.ϵ[] += (
		numaccepts * updatemethod.metro_norm - updatemethod.metro_target_acc
		) * 0.2
	return nothing
end