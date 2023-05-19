struct HeatbathUpdate <: AbstractUpdate
    MAXIT::Int64
    prefactor::Float64
    _temporary_for_staples::TemporaryField

    function HeatbathUpdate(U, MAXIT)
        prefactor = U.NC / U.β
        _temporary_for_staples = TemporaryField(U)
        return new(MAXIT, prefactor, _temporary_for_staples)
    end
end

function update!(
    updatemethod::HeatbathUpdate,
    U::Gaugefield,
    rng,
    verbose::VerboseLevel;
    metro_test = true,
)
    
    heatbath_sweep!(
        U,
        updatemethod._temporary_for_staples,
        updatemethod.MAXIT,
        updatemethod.prefactor,
        rng,
    )
    recalc_gauge_action!(U)

    return numaccepts / (U.NV * 4)
end

const to = TimerOutput()

function heatbath_sweep!(U::Gaugefield, staples::TemporaryField, MAXIT, prefactor, rng)
    NX, NY, NZ, NT = size(U)

    #staple_eachsite!(staples, U)

    for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
					for μ in 1:4
                        link = U[μ][ix,iy,iz,it]
                        #A_adj = staples[μ][ix,iy,iz,it]'
                        A_adj = staple(U, μ, site)'

                        subblock = make_submatrix(link * A_adj, 1, 2)
                        tmp = embed_into_SU3(
                            heatbath_SU2(subblock, MAXIT, prefactor, rng),
                            1, 2,
                        )
                        link = tmp * link

                        subblock = make_submatrix(link * A_adj, 1, 3)
                        tmp = embed_into_SU3(
                            heatbath_SU2(subblock, MAXIT, prefactor, rng),
                            1, 3,
                        )
                        link = tmp * link
                        
                        subblock = make_submatrix(link * A_adj, 2, 3)
                        tmp = embed_into_SU3(
                            heatbath_SU2(subblock, MAXIT, prefactor, rng),
                            2, 3,
                        )
                        U[μ][ix,iy,iz,it] = tmp * link
					end
				end
			end
		end
	end

    return nothing
end

function heatbath_SU2(A::SMatrix{2,2,ComplexF64,4}, MAXIT, prefactor, rng)
    r0 = 1
    λ2 = 1
    a_norm = 1 / sqrt(real(det(A))) # Take real of det(A) to avoid λ2 being a complex number
    V = a_norm * A
    i = 1
    
    while r0^2 + λ2 >= 1
        if i > MAXIT
            return eye2
        end

        r1 = 1 - rand(rng)
        r2 = 1 - rand(rng)
        r3 = 1 - rand(rng)

        λ2 = (-0.25 * prefactor * a_norm) * (log(r1) + cos(2π * r2)^2 * log(r3))

        r0 = rand(rng)
        i += 1
    end

    x0 = 1 - 2 * λ2
    abs_x = sqrt(1 - x0^2)

    φ = rand(rng)
    cosϑ = 1 - 2 * rand(rng)
    vec_norm = abs_x * sqrt(1 - cosϑ^2)

    x1 = vec_norm * cos(2π * φ)
    x2 = vec_norm * sin(2π * φ)
    x3 = abs_x * cosϑ
    
    mat = @SMatrix [
        x0+im*x3 x2+im*x1
        -x2+im*x1 x0-im*x3
    ]
    return mat * V'
end