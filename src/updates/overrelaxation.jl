struct ORUpdate <: AbstractUpdate
    prefactor::Float64
    _temporary_for_staples::TemporaryField

    function ORUpdate(U)
        prefactor = U.β / U.NC
        return new(prefactor)
    end
end

function update!(
    updatemethod::T,
    U::Gaugefield,
    rng,
    verbose::VerboseLevel;
    metro_test::Bool = true,
    ) where {T<:ORUpdate}
    
    prefactor = updatemethod.prefactor
    _temporary_for_staples = updatemethod._temporary_for_staples

    numaccepts = OR_sweep!(U, _temporary_for_staples, prefactor, rng)
    recalc_gauge_action!(U)

    return numaccepts / (U.NV * 4)
end

function OR_sweep!(U::Gaugefield, staples::TemporaryField, prefactor, rng)
    NX, NY, NZ, NT = size(U)
    staple_eachsite!(staples, U)
    numaccepts = 0

    for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					for μ in 1:4
                        A = staples[μ][ix,iy,iz,it]
                        old_link = U[μ][ix,iy,iz,it]

                        tmp = 1/6 * A'
                        or_mat = kenney_laub(tmp)

                        new_link = or_mat' * old_link' * or_mat'
                        ΔS = prefactor * real(tr((new_link - old_link) * A'))

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