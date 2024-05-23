const StaggeredEOPreFermionfield{B,T,A} = EvenOdd{B,T,A,1}

function calc_dSfdU!(
    dU, fermion_action::StaggeredEOPreFermionAction{4}, U, ϕ_eo::StaggeredEOPreFermionfield
)
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    X_eo, Y_eo, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    anti = D.anti_periodic

    clear!(X_eo) # initial guess is zero
    solve_dirac!(X_eo, DdagD, ϕ_eo, Y_eo, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp
    mul_oe!(Y_eo, U, X_eo, anti, true, false)
    add_staggered_eo_derivative!(dU, U, X_eo, Y_eo, anti)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::StaggeredEOPreFermionAction{Nf}, U, ϕ_eo::StaggeredEOPreFermionfield
) where {Nf}
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_md
    n = rhmc.coeffs.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    anti = D.anti_periodic
    Xs = fermion_action.rhmc_temps1[1:n+1]
    Ys = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for X in Xs
        clear!(X)
    end
    shifts = rhmc.coeffs_inverse.β
    coeffs = rhmc.coeffs_inverse.α
    solve_dirac_multishift!(Xs, shifts, DdagD, ϕ_eo, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        mul_oe!(Ys[i+1], U, Xs[i+1], anti, true, false)
        add_staggered_eo_derivative!(dU, U, Xs[i+1], Ys[i+1], anti; coeff=coeffs[i])
    end
    return nothing
end

function add_staggered_eo_derivative!(
    dU, U, X_eo::T, Y_eo::T, anti; coeff=1
) where {T<:StaggeredEOPreFermionfield}
    check_dims(dU, U, X_eo, Y_eo)
    X = X_eo.parent
    Y = Y_eo.parent
    NT = dims(U)[4]
    fac = float_type(U)(-0.5coeff)

    @batch for site in eachindex(dU)
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        add_staggered_eo_derivative_kernel!(dU, U, X, Y, site, bc⁺, fac)
    end
end

function add_staggered_eo_derivative_kernel!(dU, U, X, Y, site, bc⁺, fac)
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field 
    NX, NY, NZ, NT = dims(U)
    NV = NX * NY * NZ * NT
    _site = eo_site(site, NX, NY, NZ, NT, NV)

    _siteμ⁺ = eo_site(move(site, 1, 1, NX), NX, NY, NZ, NT, NV)
    η = staggered_η(Val(1), site)
    B = ckron(X[_siteμ⁺], Y[_site])
    C = ckron(Y[_siteμ⁺], X[_site])
    dU[1, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[1, site], B - C))

    _siteμ⁺ = eo_site(move(site, 2, 1, NY), NX, NY, NZ, NT, NV)
    η = staggered_η(Val(2), site)
    B = ckron(X[_siteμ⁺], Y[_site])
    C = ckron(Y[_siteμ⁺], X[_site])
    dU[2, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[2, site], B - C))

    _siteμ⁺ = eo_site(move(site, 3, 1, NZ), NX, NY, NZ, NT, NV)
    η = staggered_η(Val(3), site)
    B = ckron(X[_siteμ⁺], Y[_site])
    C = ckron(Y[_siteμ⁺], X[_site])
    dU[3, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[3, site], B - C))

    _siteμ⁺ = eo_site(move(site, 4, 1, NT), NX, NY, NZ, NT, NV)
    η = bc⁺ * staggered_η(Val(4), site)
    B = ckron(X[_siteμ⁺], Y[_site])
    C = ckron(Y[_siteμ⁺], X[_site])
    dU[4, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[4, site], B - C))
    return nothing
end
