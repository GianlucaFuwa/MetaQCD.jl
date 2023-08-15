struct HeatbathUpdate <: AbstractUpdate # maybe add eo as type parameter?
    eo::Bool
    MAXIT::Int64
    numHB::Int64
    numOR::Int64

    HeatbathUpdate(::Gaugefield, eo, MAXIT, numHB, numOR) = new(eo, MAXIT, numHB, numOR)
end

function update!(updatemethod::HeatbathUpdate, U, ::VerboseLevel; kwargs...)
    for _ in 1:updatemethod.numHB
        if updatemethod.eo
            heatbath_sweep_eo!(U, updatemethod.MAXIT)
        else
            heatbath_sweep!(U, updatemethod.MAXIT)
        end
    end

    numaccepts = 0.0

    for _ in 1:updatemethod.numOR
        if updatemethod.eo
            numaccepts += overrelaxation_sweep_eo!(U, metro_test=true)
        else
            numaccepts += overrelaxation_sweep!(U, metro_test=true)
        end
    end

    normalize!(U)
    U.Sg = calc_gauge_action(U)

    if updatemethod.numOR == 0
        numaccepts = 1.0
    else
        numaccepts /= U.NV * 4 * updatemethod.numOR
    end

    return numaccepts
end

function heatbath_sweep!(U, MAXIT)
    action_factor = 3 / U.β

    @inbounds for μ in 1:4
        for site in eachindex(U)
            old_link = U[μ][site]
            A = staple(U, μ, site)
            new_link = heatbath_SU3(old_link, A, MAXIT, action_factor)
            U[μ][site] = new_link
        end
	end

    return nothing
end

function heatbath_sweep_eo!(U, MAXIT)
    NX, NY, NZ, NT = size(U)
    action_factor = 3 / U.β

    for μ in 1:4
        for pass in 1:2
            @threads for it in 1:NT
                for iz in 1:NZ
                    for iy in 1:NY
                        for ix in 1+iseven(iy + iz + it + pass):2:NX
                            site = SiteCoords(ix, iy, iz, it)
                            old_link = U[μ][site]
                            A = staple(U, μ, site)
                            new_link = heatbath_SU3(old_link, A, MAXIT, action_factor)
                            U[μ][site] = new_link
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function heatbath_SU3(old_link, A, MAXIT, action_factor)
    subblock = make_submatrix(cmatmul_od(old_link, A), 1, 2)
    tmp = embed_into_SU3(heatbath_SU2(subblock, MAXIT, action_factor), 1, 2)
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_od(old_link, A), 1, 3)
    tmp = embed_into_SU3(heatbath_SU2(subblock, MAXIT, action_factor), 1, 3)
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_od(old_link, A), 2, 3)
    tmp = embed_into_SU3(heatbath_SU2(subblock, MAXIT, action_factor), 2, 3)
    new_link = cmatmul_oo(tmp, old_link)
    return new_link
end

function heatbath_SU2(A, MAXIT, action_factor)
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

        λ2 = (-0.25 * action_factor * a_norm) * (x1 + x2^2 * x3)

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
