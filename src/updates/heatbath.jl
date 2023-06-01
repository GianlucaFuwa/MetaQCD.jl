struct HeatbathUpdate <: AbstractUpdate
    eo::Bool
    MAXIT::Int64
    prefactor_HB::Float64
    prefactor_OR::Float64
    numHB::Int64
    numOR::Int64

    function HeatbathUpdate(U, eo, MAXIT, numHB, numOR)
        return new(
            eo,
            MAXIT, 
            U.NC / U.β, 
            U.β / U.NC, 
            numHB, 
            numOR,
        )
    end
end

function update!(
    updatemethod::HeatbathUpdate,
    U::Gaugefield,
    verbose::VerboseLevel;
    metro_test::Bool = true,
)
    
    for _ in 1:updatemethod.numHB
        if updatemethod.eo
            heatbath_sweep_eo!(
                U,
                updatemethod,
            )
        else
            heatbath_sweep!(
                U,
                updatemethod,
            )
        end
    end

    numaccepts = 0.0

    for _ in 1:updatemethod.numOR
        numaccepts += OR_sweep!(
            U,
            updatemethod,
            metro_test = metro_test,
        )
    end

    numaccepts = updatemethod.numOR == 0 ? 1.0 : numaccepts / (U.NV * 4 * updatemethod.numOR)
    return numaccepts
end

function heatbath_sweep!(U::Gaugefield{T}, hb) where {T}
    NX, NY, NZ, NT = size(U)
    MAXIT = hb.MAXIT
    prefactor = hb.prefactor_HB
    staple = T()

    for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
					for μ in 1:4
                        link = U[μ][ix,iy,iz,it]
                        A_adj = staple(U, μ, site)'

                        subblock = make_submatrix(link * A_adj, 1, 2)
                        tmp = embed_into_SU3(
                            heatbath_SU2(subblock, MAXIT, prefactor),
                            1, 2,
                        )
                        link = tmp * link

                        subblock = make_submatrix(link * A_adj, 1, 3)
                        tmp = embed_into_SU3(
                            heatbath_SU2(subblock, MAXIT, prefactor),
                            1, 3,
                        )
                        link = tmp * link
                        
                        subblock = make_submatrix(link * A_adj, 2, 3)
                        tmp = embed_into_SU3(
                            heatbath_SU2(subblock, MAXIT, prefactor),
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

function heatbath_sweep_eo!(U::Gaugefield{T}, hb) where {T}
    NX, NY, NZ, NT = size(U)
    MAXIT = hb.MAXIT
    prefactor = hb.prefactor_HB
    staple = T()
    for μ in 1:4
        for eo in 1:2
            @threads :static for it in 1:NT
                for iz in 1:NZ
                    for iy in 1:NY
                        offset = ((it + iz + iy) & 1) ⊻ eo
                        for ix in 1+offset:2:NX
                            site = SiteCoords(ix, iy, iz, it)
                            link = U[μ][ix,iy,iz,it]
                            A_adj = staple(U, μ, site)'

                            subblock = make_submatrix(link * A_adj, 1, 2)
                            tmp = embed_into_SU3(
                                heatbath_SU2(subblock, MAXIT, prefactor),
                                1, 2,
                            )
                            link = tmp * link

                            subblock = make_submatrix(link * A_adj, 1, 3)
                            tmp = embed_into_SU3(
                                heatbath_SU2(subblock, MAXIT, prefactor),
                                1, 3,
                            )
                            link = tmp * link
                            
                            subblock = make_submatrix(link * A_adj, 2, 3)
                            tmp = embed_into_SU3(
                                heatbath_SU2(subblock, MAXIT, prefactor),
                                2, 3,
                            )
                            U[μ][ix,iy,iz,it] = tmp * link
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function heatbath_SU2(A, MAXIT, prefactor)
    r0 = 1
    λ2 = 1
    a_norm = 1 / sqrt(real(det(A)))
    V = a_norm * A
    i = 1
    
    while r0^2 + λ2 >= 1
        if i > MAXIT
            return eye2
        end

        r1 = 1 - rand()
        r2 = 1 - rand()
        r3 = 1 - rand()

        λ2 = (-0.25 * prefactor * a_norm) * (log(r1) + cos(2π * r2)^2 * log(r3))

        r0 = rand()
        i += 1
    end

    x0 = 1 - 2 * λ2
    abs_x = sqrt(1 - x0^2)

    φ = rand()
    cosϑ = 1 - 2 * rand()
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

function OR_sweep!(U::Gaugefield{T}, hb; metro_test = true) where {T}
    NX, NY, NZ, NT = size(U)
    prefactor = hb.prefactor_OR
    numaccepts = 0
    staple = T()

    for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
					for μ in 1:4
                        A_adj = staple(U, μ, site)'
                        old_link = U[μ][ix,iy,iz,it]

                        tmp = 1/6 * A_adj
                        or_mat = kenney_laub(tmp)

                        new_link = or_mat' * old_link' * or_mat'
                        ΔS = prefactor * real(multr(new_link - old_link, A_adj))
                        accept = metro_test ? (rand() < exp(-ΔS)) : true

                        if accept
                            U[μ][ix,iy,iz,it] = new_link
                        end

                        numaccepts += accept
					end
				end
			end
		end
	end

    return numaccepts
end