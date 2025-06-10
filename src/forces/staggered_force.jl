function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,8,TD}, U, ϕ::StaggeredSpinorfield
) where {TD<:StaggeredDiracOperator}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition

    clear!(X) # initial guess is zero
    iters, res = solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp

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

    LinearAlgebra.mul!(Y, D, X)
    add_staggered_derivative!(dU, U, X, Y, bc)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{true,Nf,TD}, U, ϕ::StaggeredSpinorfield
) where {Nf,TD<:StaggeredDiracOperator}
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
        Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters
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
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1])
        add_staggered_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_derivative!(
    dU::Colorfield{CPU,T,M}, U::Gaugefield{CPU,T,M}, X::TF, Y::TF, bc; coeff=1
) where {T,M,TF<:StaggeredSpinorfield{CPU,T,M}}
    fac = T(-0.5coeff)

    @batch for site in eachindex(dU, U, X, Y)
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end

    update_halo!(dU)
    return nothing
end

function add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    NX, NY, NZ, NT = dims(U)

    # use @nexprs here to statically generate the loop
    # this makes it so Val(i) is well defined at each iteration and no type-instabilities arise
    @nexprs 4 i -> (
        siteμ⁺ = move(site, i, 1, (NX, NY, NZ, NT)[i]);
        η = staggered_η(Val(i), site);
        B = ckron(apply_bc(X[siteμ⁺], bc, site, Val(1), NT, Val(i)), Y[site]);
        C = ckron(apply_bc(Y[siteμ⁺], bc, site, Val(1), NT, Val(i)), X[site]);
        dU[i, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[i, site], B - C))
    )
    return nothing
end
