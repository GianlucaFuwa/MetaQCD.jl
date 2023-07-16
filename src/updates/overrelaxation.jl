function overrelaxation_sweep!(U::Gaugefield{GA}; metro_test = true) where {GA}
    numaccepts = 0
    staple = GA()
    action_factor = -U.β / 3

    for site in eachindex(U)
        for μ in 1:4
            A_adj = staple(U, μ, site)'
            old_link = U[μ][site]
            new_link = overrelaxation_subgroups(old_link, A_adj)

            ΔS = action_factor * real(multr(new_link - old_link, A_adj))
            accept = metro_test ? (rand() < exp(-ΔS)) : true

            if accept
                U[μ][site] = new_link
                numaccepts += accept
            end
        end
	end

    return numaccepts
end

function overrelaxation_sweep_eo!(U::Gaugefield{GA}; metro_test = true) where {GA}
    NX, NY, NZ, NT = size(U)
    spacing = 8
    numaccepts = zeros(Float64, nthreads() * spacing)
    staple = GA()
    action_factor = -U.β / 3

    for μ in 1:4
        for pass in 1:2
            @threads for it in 1:NT
                for iz in 1:NZ
                    for iy in 1:NY
                        for ix in 1+iseven(iy + iz + it + pass):2:NX
                            site = SiteCoords(ix, iy, iz, it)
                            A_adj = staple(U, μ, site)'
                            old_link = U[μ][site]
                            new_link = overrelaxation_subgroups(old_link, A_adj)

                            ΔS = action_factor * real(multr(new_link - old_link, A_adj))
                            accept = metro_test ? (rand() < exp(-ΔS)) : true

                            if accept
                                U[μ][site] = new_link
                                numaccepts[threadid() * spacing] += accept
                            end
                        end
                    end
                end
            end
        end
	end

    return sum(numaccepts)
end

function overrelaxation_kenneylaub(link, A_adj)
    tmp = 1/6 * A_adj
    or_mat = kenney_laub(tmp)
    link = cmatmul_ddd(or_mat, link, or_mat)
    return link
end

function overrelaxation_subgroups(link, A_adj)
    subblock = make_submatrix(cmatmul_oo(link, A_adj), 1, 2)
    tmp = embed_into_SU3(overrelaxation_SU2(subblock), 1, 2)
    link = cmatmul_oo(tmp, link)

    subblock = make_submatrix(cmatmul_oo(link, A_adj), 1, 3)
    tmp = embed_into_SU3(overrelaxation_SU2(subblock), 1, 3)
    link = cmatmul_oo(tmp, link)

    subblock = make_submatrix(cmatmul_oo(link, A_adj), 2, 3)
    tmp = embed_into_SU3(overrelaxation_SU2(subblock), 2, 3)
    link = cmatmul_oo(tmp, link)
    return link
end

function overrelaxation_SU2(subblock)
    a_norm = 1 / sqrt(real(det(subblock)))
    V = a_norm * subblock
    return cmatmul_dd(V, V)
end
