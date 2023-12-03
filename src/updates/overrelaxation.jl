struct Subgroups end
struct KenneyLaub end
struct Overrelaxation{ALG} end

function Overrelaxation(algorithm)
    if algorithm == "subgroups"
        ALG = Subgroups
    elseif algorithm == "kenney-laub"
        ALG = KenneyLaub
    end
    return Overrelaxation{ALG}()
end

function (or::Overrelaxation{ALG})(U, μ, site, action_factor) where {ALG}
    A_adj = staple(U, μ, site)'
    old_link = U[μ][site]
    new_link = overrelaxation_SU3(ALG(), old_link, A_adj)

    ΔSg = action_factor * real(multr(new_link - old_link, A_adj))
    accept = (rand(Float64) < exp(-ΔSg))

    if accept
        U[μ][site] = new_link
        U.Sg += ΔSg
    end
    return accept
end

function overrelaxation_SU3(::KenneyLaub, link, A_adj)
    tmp = 1/6 * A_adj
    or_mat = kenney_laub(tmp)
    link = cmatmul_ddd(or_mat, link, or_mat)
    return link
end

function overrelaxation_SU3(::Subgroups, link, A_adj)
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
