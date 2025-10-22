function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,4,TD}, U, ϕ_eo::StaggeredEOPreSpinorfield
) where {TD<:StaggeredEOPreDiracOperator}
    clear!(dU)
    X_eo, Y_eo, temp1, temp2 = fermion_action.temps[1:4]
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    solver_md = fermion_action.solver_md
    tol, maxiters, datafile = get_info(solver_md)

    clear!(X_eo) # initial guess is zero
    solve_dirac!(X_eo, DdagD, ϕ_eo, Y_eo, temp1, temp2; tol, maxiters, datafile)

    clear!(Y_eo)
    mul_oe!(Y_eo, U, X_eo, bc, true, false)
    add_staggered_eo_derivative!(dU, U, X_eo, Y_eo, bc)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{true,Nf,TD}, U, ϕ_eo::StaggeredEOPreSpinorfield
) where {Nf,TD<:StaggeredEOPreDiracOperator}
    clear!(dU)
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    temp1, temp2 = fermion_action.temps[1:2]
    Xs = fermion_action.temps[3:n+3]
    Ys = fermion_action.temps[n+4:2n+4]
    solver_md = fermion_action.solver_md
    tol, maxiters, datafile = get_info(solver_md)

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    solve_dirac_multishift!(
        Xs, shifts, DdagD, ϕ_eo, temp1, temp2, Ys; tol, maxiters, datafile
    )

    for i in 1:n
        mul_oe!(Ys[i+1], U, Xs[i+1], bc, true, false)
        add_staggered_eo_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_eo_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,TU,M}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {B,T,M,TU,TF<:StaggeredEOPreSpinorfield{B,TU,M}}
    X = X_eo.parent
    Y = Y_eo.parent
    fac = T(-0.5coeff)
    itr = eachindex(dU, U, X, Y)
    padded_bulk = dU.topology.bulk_sites_padded

    parallelfor(itr, B, Val(M), (X_eo, Y_eo), (dU,), (dU, U, X, Y)) do site, (dU, U, X, Y)
        add_staggered_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, padded_bulk, T)
    end

    return nothing
end

@inline function add_staggered_eo_derivative_kernel!(
    dU, U, X, Y, site, bc, fac, padded_bulk, ::Type{T}
) where {T}
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field
    NT = size(U, 4)
    _site = map_to_half(site, padded_bulk)

    # use @nexprs here to statically generate the loop
    # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
    @inbounds begin
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            _siteμ⁺ = map_to_half(move(site, μ, 1, Nμ), padded_bulk);
            η = staggered_η(Val(μ), site, T);
            B = ckron(apply_bc(X[_siteμ⁺], bc, site, Val(1), NT, Val(μ)), Y[_site]);
            C = ckron(apply_bc(Y[_siteμ⁺], bc, site, Val(1), NT, Val(μ)), X[_site]);
            dU[μ, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[μ, site], B - C))
        )
    end

    return nothing
end
