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

Base.eltype(::Overrelaxation{ALG}) where {ALG} = ALG

function (or::Overrelaxation{ALG})(U::Gaugefield{CPUD,T}, μ, site, action_factor) where {ALG,T}
    A_adj = staple(U, μ, site)'
    old_link = U[μ,site]
    new_link = overrelaxation_SU3(ALG(), old_link, A_adj)

    ΔSg = action_factor * real(multr(new_link - old_link, A_adj))
    accept = (rand(T) < exp(-ΔSg))

    accept && (U[μ,site] = proj_onto_SU3(new_link))
    return accept
end

function overrelaxation_SU3(::KenneyLaub, link::SMatrix{3,3,Complex{T},9}, A_adj) where {T}
    tmp = 1//6 * A_adj
    or_mat = kenney_laub(tmp)
    link = cmatmul_ddd(or_mat, link, or_mat)
    return link
end

function overrelaxation_SU3(::Subgroups, link, A_adj)
    subblock = make_submatrix_12(cmatmul_oo(link, A_adj))
    tmp = embed_into_SU3_12(overrelaxation_SU2(subblock))
    link = cmatmul_oo(tmp, link)

    subblock = make_submatrix_13(cmatmul_oo(link, A_adj))
    tmp = embed_into_SU3_13(overrelaxation_SU2(subblock))
    link = cmatmul_oo(tmp, link)

    subblock = make_submatrix_23(cmatmul_oo(link, A_adj))
    tmp = embed_into_SU3_23(overrelaxation_SU2(subblock))
    link = cmatmul_oo(tmp, link)
    return link
end

@inline function overrelaxation_SU2(subblock)
    a_norm = 1 / sqrt(real(det(subblock)))
    V = a_norm * subblock
    return cmatmul_dd(V, V)
end
