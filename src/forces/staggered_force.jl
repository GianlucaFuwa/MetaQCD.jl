const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

function calc_dSfdU!(
    dU, fermion_action::StaggeredFermionAction{4}, U, ψ::StaggeredFermionfield
)
    @assert dims(dU) == dims(U) == dims(ψ)
    X, Y, temp1, temp2 = get_cg_temps(fermion_action)
    D = fermion_action.D
    replace_U!(D, U)
    DdagD = DdaggerD(D)
    clear!(X) # initial guess is zero
    solve_D⁻¹x!(X, DdagD, ψ, Y, temp1, temp2)
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    TA_from_XY!(dU, U, X, Y, D.anti_periodic)
    return nothing
end

function TA_from_XY!(dU, U, X::T, Y::T, anti) where {T<:StaggeredFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)
    monehalf = float_type(U)(-0.5)

    for site in eachindex(dU)
        siteμ⁺ = move(site, 1, 1, NX)
        η = staggered_η(Val(1), site)
        temp = ckron(X[siteμ⁺], Y[site]) - ckron(Y[siteμ⁺], X[site])
        dU[1, site] = monehalf * η * traceless_antihermitian(cmatmul_oo(U[1, site], temp))

        siteμ⁺ = move(site, 2, 1, NY)
        η = staggered_η(Val(2), site)
        temp = ckron(X[siteμ⁺], Y[site]) - ckron(Y[siteμ⁺], X[site])
        dU[2, site] = monehalf * η * traceless_antihermitian(cmatmul_oo(U[2, site], temp))

        siteμ⁺ = move(site, 3, 1, NZ)
        η = staggered_η(Val(3), site)
        temp = ckron(X[siteμ⁺], Y[site]) - ckron(Y[siteμ⁺], X[site])
        dU[3, site] = monehalf * η * traceless_antihermitian(cmatmul_oo(U[3, site], temp))

        siteμ⁺ = move(site, 4, 1, NT)
        η = staggered_η(Val(4), site)
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        temp = ckron(bc⁺ * X[siteμ⁺], Y[site]) - ckron(bc⁺ * Y[siteμ⁺], X[site])
        dU[4, site] = monehalf * η * traceless_antihermitian(cmatmul_oo(U[4, site], temp))
    end
end
