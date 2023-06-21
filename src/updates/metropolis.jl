struct MetroUpdate{TG} <: AbstractUpdate
	ϵ::Base.RefValue{Float64}
	multi_hit::Int64
	target_acc::Float64
	mnorm::Float64
	meta_enabled::Bool

	function MetroUpdate(
		U::Gaugefield{TG},
		ϵ,
		multi_hit,
		target_acc,
		meta_enabled,
	) where {TG}
		mnorm = 1 / (U.NV * 4 * multi_hit)
		m_ϵ = Base.RefValue{Float64}(ϵ)

        if meta_enabled == true
            @error "Metadynamics is not supported for metropolis yet"
        end

		return new{TG}(m_ϵ, multi_hit, target_acc, mnorm, meta_enabled)
	end
end

function update!(
	updatemethod,
	U,
	verbose::VerboseLevel;
	Bias = nothing,
	metro_test = true,
)
	if updatemethod.meta_enabled
		numaccepts = metro_sweep_meta!(U, Bias, updatemethod, metro_test = true)
	else
		numaccepts = metro_sweep!(U, updatemethod, metro_test = true)
	end

    normalize!(U)
    adjust_ϵ!(updatemethod, numaccepts)
	return numaccepts * updatemethod.mnorm
end

function metro_sweep!(
	U::Gaugefield{T},
	metro;
	metro_test = true,
) where {T}
	NX, NY, NZ, NT = size(U)
	β = U.β
	ϵ = metro.ϵ[]
	multi_hit = metro.multi_hit
	numaccept = 0
	staple = T()

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)
					for μ in 1:4

						for _ in 1:multi_hit
							X = gen_SU3_matrix(ϵ)
							link = U[μ][ix,iy,iz,it]
                            XU = cmatmul_oo(X, link)

							A_adj = staple(U, μ, site)'

							ΔSg = β/3 * real(multr((XU - link), A_adj))

							accept = metro_test ? (rand() ≤ exp(-ΔSg)) : true

							if accept
								U.Sg += ΔSg
								U[μ][site] = X * U[μ][site]
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

function metro_sweep_meta!( # TODO
	U::Gaugefield{T},
	Bias,
	metro;
	metro_test = true,
) where {T}
	NX, NY, NZ, NT = size(U)
	β = U.β
	ϵ = metro.ϵ[]
	multi_hit = metro.multi_hit
	numaccept = 0
	staple = T()

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)

					for μ in 1:4

						for _ in 1:multi_hit
							X = gen_SU3_matrix(ϵ)
							link = U[μ][ix,iy,iz,it]
                            XU = cmatmul_oo(X, link)

							A_adj = staple(U, μ, site)'

							CV = U.CV
							ΔSg = β/3 * real(multr((XU - link), A_adj))
							ΔCV = nothing # TODO
							ΔV = Bias(CV + ΔCV) - Bias(CV)

							accept = metro_test ? (rand() ≤ exp(-ΔSg - ΔV)) : true

							if accept
								U.Sg += ΔSg
								U.CV += ΔCV
								U[μ][site] = X * U[μ][site]
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

function adjust_ϵ!(metro, numaccepts)
	metro.ϵ[] += (numaccepts * metro.mnorm - metro.target_acc) * 0.2
    metro.ϵ[] = min(1.0, metro.ϵ[])
	return nothing
end
