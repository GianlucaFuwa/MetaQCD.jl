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

    @batch for site in eachindex(dU)
        site1⁺ = move(site, 1, 1, NX)
        Bₙ₁ = ckron_sum(
            cmvmul_spin_proj(U[1, site], X[site1⁺], Val(-1), Val(false)), Y[site]
        )
        Cₙ₁ = ckron_sum(
            Y[site1⁺], cmvmul_spin_proj(U[1, site], X[site], Val(1), Val(false))
        )
        dU[1, site] = traceless_antihermitian(Bₙ₁ - Cₙ₁)

        site2⁺ = move(site, 1, 1, NY)
        Bₙ₂ = ckron_sum(
            cmvmul_spin_proj(U[2, site], X[site2⁺], Val(-2), Val(false)), Y[site]
        )
        Cₙ₂ = ckron_sum(
            Y[site2⁺], cmvmul_spin_proj(U[2, site], X[site], Val(2), Val(false))
        )
        dU[2, site] = traceless_antihermitian(Bₙ₂ - Cₙ₂)

        site3⁺ = move(site, 1, 1, NZ)
        Bₙ₃ = ckron_sum(
            cmvmul_spin_proj(U[3, site], X[site3⁺], Val(-3), Val(false)), Y[site]
        )
        Cₙ₃ = ckron_sum(
            Y[site3⁺], cmvmul_spin_proj(U[3, site], X[site], Val(3), Val(false))
        )
        dU[3, site] = traceless_antihermitian(Bₙ₃ - Cₙ₃)

        site4⁺ = move(site, 1, 1, NT)
        Bₙ₄ = ckron_sum(
            cmvmul_spin_proj(U[4, site], X[site4⁺], Val(-4), Val(false)), Y[site]
        )
        Cₙ₄ = ckron_sum(
            Y[site4⁺], cmvmul_spin_proj(U[4, site], X[site], Val(4), Val(false))
        )
        dU[4, site] = traceless_antihermitian(Bₙ₄ - Cₙ₄)
    end
end
