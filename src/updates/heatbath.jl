struct Heatbath_update <: AbstractUpdate
    MAXIT::Int64
    prefactor::Float64

    function Heatbath_update(U, MAXIT)
        prefactor = U.NC / U.β
        return new(MAXIT, prefactor)
    end
end

get_MAXIT(hb::Heatbath_update) = hb.MAXIT
get_prefactor(hb::Heatbath_update) = hb.prefactor

function update!(
    updatemethod::T,
    U::Gaugefield,
    rng::Xoshiro,
    verbose::Verbose_level;
    metro_test::Bool = true,
    ) where {T<:Heatbath_update}
    
    MAXIT = get_MAXIT(updatemethod)
    prefactor = get_prefactor(updatemethod)
    heatbath_sweep!(U, MAXIT, prefactor, rng)
    recalc_GaugeAction!(U)

    return numaccepts / U.NV / 4.0
end

function heatbath_sweep!(U::Gaugefield, MAXIT, prefactor, rng::Xoshiro)
    NX, NY, NZ, NT = size(U)

    for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)
					for μ = 1:4
                        A = staple(U, μ, site)
                        UA = U[μ][ix,iy,iz,it] * A'

                        subblock = SU2_from_SU3(UA, 1)
                        subblock = heatbath_SU2(subblock, MAXIT, prefactor, rng) 
                        R = SU3_from_SU2(subblock, 1)

                        subblock = SU2_from_SU3(UA, 2)
                        subblock = heatbath_SU2(subblock, MAXIT, prefactor, rng) 
                        S = SU3_from_SU2(subblock, 2)

                        subblock = SU2_from_SU3(UA, 3)
                        subblock = heatbath_SU2(subblock, MAXIT, prefactor, rng) 
                        T = SU3_from_SU2(subblock, 3)

                        U[μ][ix,iy,iz,it] = T * S * R * U[μ][ix,iy,iz,it]
					end
				end
			end
		end
	end
    return nothing
end

function heatbath_SU2(A::SMatrix{2,2,ComplexF64,4}, MAXIT, prefactor, rng)
    r0 = 1.0
    λ2 = 1.0
    a = real(sqrt(det(A)))
    i = 1
    while r0^2 + λ2 > 1.0
        if i > MAXIT
            return SMatrix{2,2,ComplexF64}([
                1.0 0
                0 1.0
            ])
        end

        r1 = 1.0 - rand(rng)
        x1 = log(r1)
        r2 = 1.0 - rand(rng)
        x2 = cos(2.0*pi*r2)^2
        r3 = 1.0 - rand(rng)
        x3 = log(r3)

        λ2 = -0.25*prefactor / a * (x1 + x2*x3)

        r0 = rand(rng)
        i += 1
    end

    x0 = 1.0 - 2.0*λ2
    absx = 1.0 - x0^2

    φ = rand(rng)
    cosϑ = 1.0 - 2.0*rand(rng)
    vec_norm = absx * sqrt(1.0 - cosϑ^2)

    x1 = vec_norm * cos(2.0*π*φ)
    x2 = vec_norm * sin(2.0*π*φ)
    x3 = absx * cosϑ
    mat = SMatrix{2,2,ComplexF64}([
        x0+im*x3  x2+im*x1
        -x2+im*x1 x0-im*x3
    ])
    return mat * A/a
end


