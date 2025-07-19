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
    dU::Colorfield{B,T}, U::Gaugefield{B,T,M}, X::TF, Y::TF, bc; coeff=1
) where {B,T,M,TF<:StaggeredSpinorfield{B,T,M}}
    fac = T(-0.5coeff)
    itr = eachindex(dU, U, X, Y)

    parallelfor(itr, B, Val(M), (X, Y), (dU,), (dU, U, X, Y)) do site, dU, U, X, Y
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end

    return nothing
end

function add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    NT = size(U, 4)

    # use @nexprs here to statically generate the loop
    # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
    @nexprs 4 μ -> (
        Nμ = axes(U, μ);
        siteμ⁺ = move(site, μ, 1, Nμ);
        η = staggered_η(Val(μ), site);
        B = ckron(apply_bc(X[siteμ⁺], bc, site, Val(1), NT, Val(μ)), Y[site]);
        C = ckron(apply_bc(Y[siteμ⁺], bc, site, Val(1), NT, Val(μ)), X[site]);
        dU[μ, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[μ, site], B - C))
    )
    return nothing
end
