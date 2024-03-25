const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

function calc_dSfdU!(
    dU, fermion_action::StaggeredFermionAction{4}, U, ψ::StaggeredFermionfield
)
    @assert dims(dU) == dims(U) == dims(ψ)
    X, Y, temp1, temp2 = get_cg_temps(fermion_action)
    D = fermion_action.D
    replace_U!(D, U)
    DdagD = DdaggerD(D)
    solve_D⁻¹x!(X, DdagD, ψ, Y, temp1, temp2)
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    TA_from_XY!(dU, U, X, Y)
    return nothing
end

function TA_from_XY!(dU, U, X::T, Y::T) where {T<:StaggeredFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)
    onehalf = float_type(U)(0.5)

    @batch for site in eachindex(dU)
        siteμ⁺ = move(site, 1, 1, NX)
        η = onehalf * staggered_η(Val(1), site)
        Bₙ = η * ckron(cmvmul(U[1, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[1, site], Y[1, siteμ⁺]))
        dU[1, site] = traceless_antihermitian(Bₙ - Cₙ)

        siteμ⁺ = move(site, 1, 1, NY)
        η = onehalf * staggered_η(Val(2), site)
        Bₙ = η * ckron(cmvmul(U[2, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[2, site], Y[1, siteμ⁺]))
        dU[2, site] = traceless_antihermitian(Bₙ - Cₙ)

        siteμ⁺ = move(site, 1, 1, NZ)
        η = onehalf * staggered_η(Val(3), site)
        Bₙ = η * ckron(cmvmul(U[3, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[3, site], Y[1, siteμ⁺]))
        dU[3, site] = traceless_antihermitian(Bₙ - Cₙ)

        siteμ⁺ = move(site, 1, 1, NT)
        η = onehalf * staggered_η(Val(4), site)
        Bₙ = η * ckron(cmvmul(U[4, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[4, site], Y[1, siteμ⁺]))
        dU[4, site] = traceless_antihermitian(Bₙ - Cₙ)
    end
end
