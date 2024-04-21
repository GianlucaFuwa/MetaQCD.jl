const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

function calc_dSfdU!(
    dU, fermion_action::StaggeredFermionAction{8}, U, ϕ::StaggeredFermionfield
)
    @assert dims(dU) == dims(U) == dims(ϕ)
    X = fermion_action.cg_x
    Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D
    replace_U!(D, U)
    DdagD = DdaggerD(D)
    clear!(X) # initial guess is zero
    solve_D⁻¹x!(X, DdagD, ϕ, Y, temp1, temp2) # Y is used here merely as a temp
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    staggered_derivative!(dU, U, X, Y, D.anti_periodic)
    return nothing
end

function staggered_derivative!(dU, U, X::T, Y::T, anti) where {T<:StaggeredFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)
    moh = float_type(U)(-0.5)

    for site in eachindex(dU)
        siteμ⁺ = move(site, 1, 1, NX)
        η = staggered_η(Val(1), site)
        B = ckron(X[siteμ⁺], Y[site])
        C = ckron(Y[siteμ⁺], X[site])
        dU[1, site] = (moh * η) * traceless_antihermitian(cmatmul_oo(U[1, site], B - C))

        siteμ⁺ = move(site, 2, 1, NY)
        η = staggered_η(Val(2), site)
        B = ckron(X[siteμ⁺], Y[site])
        C = ckron(Y[siteμ⁺], X[site])
        dU[2, site] = (moh * η) * traceless_antihermitian(cmatmul_oo(U[2, site], B - C))

        siteμ⁺ = move(site, 3, 1, NZ)
        η = staggered_η(Val(3), site)
        B = ckron(X[siteμ⁺], Y[site])
        C = ckron(Y[siteμ⁺], X[site])
        dU[3, site] = (moh * η) * traceless_antihermitian(cmatmul_oo(U[3, site], B - C))

        siteμ⁺ = move(site, 4, 1, NT)
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        η = bc⁺ * staggered_η(Val(4), site)
        B = ckron(X[siteμ⁺], Y[site])
        C = ckron(Y[siteμ⁺], X[site])
        dU[4, site] = (moh * η) * traceless_antihermitian(cmatmul_oo(U[4, site], B - C))
    end
end
