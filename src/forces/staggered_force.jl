const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

function calc_dSfdU!(
    dU, fermion_action::StaggeredFermionAction{8}, U, ϕ::StaggeredFermionfield
)
    @assert dims(dU) == dims(U) == dims(ϕ)
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)

    clear!(X) # initial guess is zero
    solve_D⁻¹x!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    add_staggered_derivative!(dU, U, X, Y, D.anti_periodic)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::StaggeredFermionAction{Nf}, U, ϕ::StaggeredFermionfield
) where {Nf}
    @assert dims(dU) == dims(U) == dims(ϕ)
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_md
    n = rhmc.coeffs.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    Xs = fermion_action.rhmc_temps1
    Ys = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for v in Xs
        clear!(v)
    end
    shifts = rhmc.coeffs_inverse.β
    coeffs = rhmc.coeffs_inverse.α
    solve_D⁻¹x_multishift!(Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1]) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        add_staggered_derivative!(dU, U, Xs[i+1], Ys[i+1], D.anti_periodic; coeff=coeffs[i])
    end
    return nothing
end

function add_staggered_derivative!(
    dU, U, X::T, Y::T, anti; coeff=1
) where {T<:StaggeredFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)
    fac = float_type(U)(-0.5coeff)

    for site in eachindex(dU)
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
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        η = bc⁺ * staggered_η(Val(4), site)
        B = ckron(X[siteμ⁺], Y[site])
        C = ckron(Y[siteμ⁺], X[site])
        dU[4, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[4, site], B - C))
    end
end
