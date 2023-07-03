struct HeatbathUpdate <: AbstractUpdate # maybe add eo as type parameter?
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
            3 / U.β,
            U.β / 3,
            numHB,
            numOR,
        )
    end
end

function update!(
    updatemethod::HeatbathUpdate,
    U,
    verbose::VerboseLevel;
    Bias = nothing,
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
        if updatemethod.eo
            numaccepts += OR_sweep_eo!(
                U,
                updatemethod,
                metro_test = metro_test,
            )
        else
            numaccepts += OR_sweep!(
                U,
                updatemethod,
                metro_test = metro_test,
            )
        end
    end

    if Bias !== nothing
        calc_smearedU!(Bias.smearing, U)
        fully_smeared_U = Bias.smearing.Usmeared_multi[end]
        CV_new = top_charge(fully_smeared_U, Bias.kind_of_cv)
        U.CV = CV_new
    end

    normalize!(U)
    numaccepts = updatemethod.numOR == 0 ? 1.0 : numaccepts / (U.NV * 4 * updatemethod.numOR)
    return numaccepts
end

function heatbath_sweep!(U::Gaugefield{GA}, hb) where {GA}
    MAXIT = hb.MAXIT
    prefactor = hb.prefactor_HB
    staple = GA()

    for site in eachindex(U)
        @inbounds for μ in 1:4
            old_link = U[μ][site]
            A = staple(U, μ, site)
            new_link = heatbath_SU3(old_link, A, MAXIT, prefactor)
            U[μ][site] = new_link
        end
	end

    return nothing
end

function heatbath_sweep_eo!(U::Gaugefield{GA}, hb) where {GA}
    MAXIT = hb.MAXIT
    prefactor = hb.prefactor_HB
    staple = GA()

    @threads :static for site in eachindex(u)
        @inbounds for μ in 1:4
            for sublattice in 1:4
                if mod1(ix + iy + iz + it + site[μ], 4) == sublattice
                    old_link = U[μ][site]
                    A = staple(U, μ, site)
                    new_link = heatbath_SU3(old_link, A, MAXIT, prefactor)
                    U[μ][site] = new_link
                end
            end
        end
    end

    return nothing
end

function heatbath_SU3(old_link, A, MAXIT, prefactor)
    subblock = make_submatrix(cmatmul_od(old_link, A), 1, 2)
    tmp = embed_into_SU3(
        heatbath_SU2(subblock, MAXIT, prefactor),
        1, 2,
    )
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_od(old_link, A), 1, 3)
    tmp = embed_into_SU3(
        heatbath_SU2(subblock, MAXIT, prefactor),
        1, 3,
    )
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_od(old_link, A), 2, 3)
    tmp = embed_into_SU3(
        heatbath_SU2(subblock, MAXIT, prefactor),
        2, 3,
    )
    new_link = cmatmul_oo(tmp, old_link)
    return new_link
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
        x1 = log(r1)
        r2 = 1 - rand()
        x2 = cos(2π * r2)
        r3 = 1 - rand()
        x3 = log(r3)

        λ2 = (-0.25 * prefactor * a_norm) * (x1 + x2^2 * x3)

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
    return cmatmul_od(mat, V)
end

function OR_sweep!(U::Gaugefield{GA}, hb; metro_test = true) where {GA}
    prefactor = hb.prefactor_OR
    numaccepts = 0
    staple = GA()

    for site in eachindex(U)
        @inbounds for μ in 1:4
            A_adj = staple(U, μ, site)'
            old_link = U[μ][site]
            new_link = overrelaxation_subgroups(old_link, A_adj)

            ΔS = prefactor * real(multr(new_link - old_link, A_adj))
            accept = metro_test ? (rand() < exp(-ΔS)) : true

            if accept
                U[μ][site] = new_link
            end

            numaccepts += accept
        end
	end

    return numaccepts
end

function OR_sweep_eo!(U::Gaugefield{GA}, hb; metro_test = true) where {GA}
    prefactor = hb.prefactor_OR
    numaccepts = 0
    staple = GA()

    for sublattice in 1:4
        @batch for site in eachindex(U)
            for μ in 1:4
                if mod1(ix + iy + iz + it + site[μ], 4) == sublattice
                    A_adj = staple(U, μ, site)'
                    old_link = U[μ][site]
                    new_link = overrelaxation_kenneylaub(old_link, A_adj)

                    ΔS = prefactor * real(multr(new_link - old_link, A_adj))
                    accept = metro_test ? (rand() < exp(-ΔS)) : true

                    if accept
                        U[μ][site] = new_link
                    end

                    numaccepts += accept
                end
            end
        end
	end

    return numaccepts
end

function overrelaxation_kenneylaub(old_link, A_adj)
    tmp = 1/6 * A_adj
    or_mat = kenney_laub(tmp)
    new_link = cmatmul_ddd(or_mat, old_link, or_mat)
    return new_link
end

function overrelaxation_subgroups(old_link, A_adj)
    subblock = make_submatrix(cmatmul_oo(old_link, A_adj), 1, 2)
    tmp = embed_into_SU3(
        overrelaxation_SU2(subblock),
        1, 2,
    )
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_oo(old_link, A_adj), 1, 3)
    tmp = embed_into_SU3(
        overrelaxation_SU2(subblock),
        1, 3,
    )
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_oo(old_link, A_adj), 2, 3)
    tmp = embed_into_SU3(
        overrelaxation_SU2(subblock),
        2, 3,
    )
    new_link = cmatmul_oo(tmp, old_link)
    return new_link
end

function overrelaxation_SU2(subblock)
    a_norm = 1 / sqrt(real(det(subblock)))
    V = a_norm * subblock
    return cmatmul_dd(V, V)
end
