struct MetroUpdate <: AbstractUpdate
	ϵ::Base.RefValue{Float64}
	multi_hit::Int64
	target_acc::Float64
	mnorm::Float64
	meta_enabled::Bool
	_temp_for_staples::TemporaryField

	function MetroUpdate(U::Gaugefield, ϵ, multi_hit, target_acc, meta_enabled)
		mnorm = 1 / (U.NV * 4 * multi_hit)
		m_ϵ = Base.RefValue{Float64}(ϵ)
		_temp_for_staples = TemporaryField(U)
		return new(m_ϵ, multi_hit, target_acc, mnorm, meta_enabled, _temp_for_staples)
	end
end

function update!(
	updatemethod::MetroUpdate,
	U::Gaugefield,
	verbose::VerboseLevel;
	Bias = nothing,
	metro_test = true,
)
	if updatemethod.meta_enabled
		numaccepts = metro_sweep_meta!(U, Bias, updatemethod, metro_test = metro_test)
	else
		numaccepts = metro_sweep!(U, updatemethod, metro_test = metro_test)
	end

	adjust_ϵ!(updatemethod, numaccepts)
	return numaccepts * updatemethod.mnorm
end

function metro_sweep!(
	U::Gaugefield,
	metro::MetroUpdate,
	metro_test = true,
)
	NX, NY, NZ, NT = size(U)
	β = U.β
	ϵ = metro.ϵ[]
	multi_hit = metro.multi_hit
	numaccept = 0

	A = metro._temp_for_staples
	staple_eachsite!(A, U)

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					for μ in 1:4

						for i in 1:multi_hit
							X = gen_SU3_matrix(ϵ)
							link = U[μ][ix,iy,iz,it]
							A_adj = A[μ][ix,iy,iz,it]'
							ΔSg = β/3 * real(tr((X * link - link) * A_adj))

							accept = metro_test ? (rand() ≤ exp(-ΔSg)) : 1
					
							if accept
								U.Sg += ΔSg
								U[μ][origin] = X * U[μ][origin]
							end

							numaccept += accept
						end

					end
				end
			end
		end
	end

	return numaccept
end

function metro_sweep_meta!(
	U::Gaugefield,
	Bias::BiasPotential,
	metro::MetroUpdate,
	metro_test = true,
)
	NX, NY, NZ, NT = size(U)
	β = U.β
	ϵ = metro.ϵ[]
	multi_hit = metro.multi_hit
	numaccept = 0

	A = metro._temp_for_staples
	staple_eachsite!(A, U)

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					for μ in 1:4
						
						for hit in 1:multi_hit
							X = gen_SU3_matrix(ϵ)
							link = U[μ][ix,iy,iz,it]
							A_adj = A[μ][ix,iy,iz,it]'

							CV = U.CV
							ΔSg = β/3 * real(tr((X * link - link) * A_adj))
							ΔCV = 1/2π * imag(tr((X * link - link) * A_adj)) 
							ΔV = Bias(CV + ΔCV) - Bias(CV)

							accept = metro_test ? (rand() ≤ exp(-ΔSg - ΔV)) : 1
					
							if accept
								U.Sg += ΔSg
								U.CV += ΔV
								U[μ][origin] = X * U[μ][origin]
							end

							numaccept += accept
						end

					end
				end
			end
		end
	end

	return numaccepts
end

function adjust_ϵ!(metro::MetroUpdate, numaccepts)
	metro.ϵ[] += (numaccepts * metro.mnorm - metro.target_acc) * 0.2
	return nothing
end