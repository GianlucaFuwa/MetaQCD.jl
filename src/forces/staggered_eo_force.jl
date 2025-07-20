function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,4,TD}, U, ϕ_eo::StaggeredEOPreSpinorfield
) where {TD<:StaggeredEOPreDiracOperator}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X_eo, Y_eo, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition

    clear!(X_eo) # initial guess is zero
    iters, res = solve_dirac!(X_eo, DdagD, ϕ_eo, Y_eo, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp

    cg_datafile = fermion_action.cg_datafile

    if cg_datafile != ""
        set_ext!(cg_datafile, MPI_INSTANCE[])
        fp = fopen(cg_datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        printf(fp, "%s", "# force")
        newline(fp)
        fclose(fp)
    end

    clear!(Y_eo)
    mul_oe!(Y_eo, U, X_eo, bc, true, false)
    add_staggered_eo_derivative!(dU, U, X_eo, Y_eo, bc)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{true,Nf,TD}, U, ϕ_eo::StaggeredEOPreSpinorfield
) where {Nf,TD<:StaggeredEOPreDiracOperator}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    Xs = fermion_action.rhmc_temps1
    Ys = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    iters, res = solve_dirac_multishift!(
        Xs, shifts, DdagD, ϕ_eo, temp1, temp2, Ys, cg_tol, cg_maxiters
    )

    cg_datafile = fermion_action.cg_datafile

    if cg_datafile != ""
        set_ext!(cg_datafile, MPI_INSTANCE[])
        fp = fopen(cg_datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        printf(fp, "%s", "# force")
        newline(fp)
        fclose(fp)
    end

    for i in 1:n
        mul_oe!(Ys[i+1], U, Xs[i+1], bc, true, false)
        add_staggered_eo_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_eo_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T,M}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {B,T,M,TF<:StaggeredEOPreSpinorfield{B,T,M}}
    X = X_eo.parent
    Y = Y_eo.parent
    fac = T(-0.5coeff)
    bulk = eachindex(dU)
    itr = eachindex(dU, U, X, Y)

    parallelfor(itr, B, Val(M), (X_eo, Y_eo), (dU,), (dU, U, X, Y)) do site, (dU, U, X, Y)
        add_staggered_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, bulk)
    end

    return nothing
end

function add_staggered_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, bulk)
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field
    NT = size(U, 4)
    _site = map_to_half(site, bulk)

    # use @nexprs here to statically generate the loop
    # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
    @nexprs 4 μ -> (
        Nμ = axes(U, μ);
        _siteμ⁺ = map_to_half(move(site, μ, 1, Nμ), bulk);
        η = staggered_η(Val(μ), site);
        B = ckron(apply_bc(X[_siteμ⁺], bc, site, Val(1), NT, Val(μ)), Y[_site]);
        C = ckron(apply_bc(Y[_siteμ⁺], bc, site, Val(1), NT, Val(μ)), X[_site]);
        dU[μ, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[μ, site], B - C))
    )
    return nothing
end
