const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

function calc_dSfdU!(dU, fermion_action::WilsonFermionAction, U, ψ::WilsonFermionfield)
    @assert dims(dU) == dims(U) == dims(ψ)
    X, Y, temp1, temp2 = get_cg_temps(fermion_action)
    D = fermion_action.D
    replace_U!(D, U)
    DdagD = DdaggerD(D)
    solve_D⁻¹x!(X, DdagD, ψ, Y, temp1, temp2)
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    return TA_from_XY!(dU, U, X, Y)
end

function TA_from_XY!(dU, U, X::T, Y::T) where {T<:WilsonFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)

    for site in eachindex(dU)
        siteμ⁺ = move(site, 1, 1, NX)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-1)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(1)), X[site])
        dU[1, site] = traceless_antihermitian(cmatmul_oo(U[1, site], B - C))

        siteμ⁺ = move(site, 2, 1, NY)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-2)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(2)), X[site])
        dU[2, site] = traceless_antihermitian(cmatmul_oo(U[2, site], B - C))

        siteμ⁺ = move(site, 3, 1, NZ)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-3)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(3)), X[site])
        dU[3, site] = traceless_antihermitian(cmatmul_oo(U[3, site], B - C))

        siteμ⁺ = move(site, 4, 1, NT)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-4)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(4)), X[site])
        dU[4, site] = traceless_antihermitian(cmatmul_oo(U[4, site], B - C))
    end
end
