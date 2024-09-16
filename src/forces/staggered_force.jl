const StaggeredSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,1}

function calc_dSfdU!(
    dU, fermion_action::StaggeredFermionAction{8}, U, ϕ::StaggeredSpinorfield
)
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition

    clear!(X) # initial guess is zero
    solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp
    LinearAlgebra.mul!(Y, D, X)
    add_staggered_derivative!(dU, U, X, Y, bc)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::StaggeredFermionAction{Nf}, U, ϕ::StaggeredSpinorfield
) where {Nf}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    rhmc = fermion_action.rhmc_info_md
    n = get_n(rhmc)
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
    solve_dirac_multishift!(Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1])
        add_staggered_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_derivative!(
    dU::Colorfield{CPU,T,M}, U::Gaugefield{CPU,T,M}, X::TF, Y::TF, bc; coeff=1
) where {T,M,TF<:StaggeredSpinorfield{CPU,T,M}}
    check_dims(dU, U, X, Y)
    fac = T(-0.5coeff)

    @batch for site in eachindex(dU)
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end

    update_halo!(dU)
    return nothing
end

function add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    NX, NY, NZ, NT = dims(U)
    siteμ⁺ = move(site, 1, 1, NX)
    η = staggered_η(Val(1), site)
    B = ckron(X[siteμ⁺], Y[site])
    C = ckron(Y[siteμ⁺], X[site])
    dU[1, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[1, site], B - C))

    siteμ⁺ = move(site, 2, 1, NY)
    η = staggered_η(Val(2), site)
    B = ckron(X[siteμ⁺], Y[site])
    C = ckron(Y[siteμ⁺], X[site])
    dU[2, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[2, site], B - C))

    siteμ⁺ = move(site, 3, 1, NZ)
    η = staggered_η(Val(3), site)
    B = ckron(X[siteμ⁺], Y[site])
    C = ckron(Y[siteμ⁺], X[site])
    dU[3, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[3, site], B - C))

    siteμ⁺ = move(site, 4, 1, NT)
    η = staggered_η(Val(4), site)
    B = ckron(apply_bc(X[siteμ⁺], bc, site, Val(1), NT), Y[site])
    C = ckron(apply_bc(Y[siteμ⁺], bc, site, Val(1), NT), X[site])
    dU[4, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[4, site], B - C))
    return nothing
end
