struct OR_update <: AbstractUpdate
    prefactor::Float64

    function OR_update(U)
        prefactor = U.β / U.NC
        return new(prefactor)
    end
end

get_prefactor(or::OR_update) = or.prefactor

function update!(
    updatemethod::T,
    U::Gaugefield,
    rng::Xoshiro,
    verbose::Verbose_level;
    metro_test::Bool = true,
    ) where {T<:OR_update}
    
    prefactor = get_prefactor(updatemethod)
    numaccepts = OR_sweep!(U, prefactor, rng)
    recalc_GaugeAction!(U)

    return numaccepts / U.NV / 4.0
end

function OR_sweep!(U::Gaugefield, prefactor, rng)
    NX, NY, NZ, NT = size(U)

    numaccepts = 0
    for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)
					for μ = 1:4
                        A = staple(U, μ, site)

                        old_link = U[μ][ix,iy,iz,it]
                        tmp = 1/6 * A'
                        or_mat = KenneyLaub(tmp)

                        new_link = or_mat' * old_link' * or_mat'
                        ΔS = prefactor * real(tr((new_link - old_link)*A'))
                        if rand(rng) < exp(-ΔS)
                            U[μ][ix,iy,iz,it] = new_link
                            numaccepts += 1
                        end
					end
				end
			end
		end
	end
    return numaccepts
end