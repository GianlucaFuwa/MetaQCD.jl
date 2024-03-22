const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

function calc_dSfdU!( # Unrooted / Nf=4
    dU,
    U,
    ψ::StaggeredFermionfield,
    fermion_action::StaggeredFermionAction{4},
)
    @assert dims(dU) == dims(U) == dims(ψ)
    X = fermion_action.temp1
    Y = fermion_action.temp2
    temps_for_cg = get_cg_temps(fermion_action)
    D = fermion_action.D(U)
    DdagD = Hermitian(D)
    solve_D⁻¹x!(X, DdagD, ψ, temps_for_cg...)
    mul!(Y, D, X)
    TA_from_XY!(dU, U, X, Y)
end

function TA_from_XY!(dU, U, X::T, Y::T, α=1) where {T<:StaggeredFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)
    coeff = float_type(U)(0.5α)

    @batch for site in eachindex(dU)
        siteμ⁺ = move(site, 1, 1, NX)
        η = coeff * staggered_η(Val(1), site)
        Bₙ = η * ckron(cmvmul(U[1, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[1, site], Y[1, siteμ⁺]))
        dU[1, site] = traceless_antihermitian(Bₙ - Cₙ)

        siteμ⁺ = move(site, 1, 1, NY)
        η = coeff * staggered_η(Val(2), site)
        Bₙ = η * ckron(cmvmul(U[2, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[2, site], Y[1, siteμ⁺]))
        dU[2, site] = traceless_antihermitian(Bₙ - Cₙ)

        siteμ⁺ = move(site, 1, 1, NZ)
        η = coeff * staggered_η(Val(3), site)
        Bₙ = η * ckron(cmvmul(U[3, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[3, site], Y[1, siteμ⁺]))
        dU[3, site] = traceless_antihermitian(Bₙ - Cₙ)

        siteμ⁺ = move(site, 1, 1, NT)
        η = coeff * staggered_η(Val(4), site)
        Bₙ = η * ckron(cmvmul(U[4, site], X[1, siteμ⁺]), Y[1, site])
        Cₙ = η * ckron(X[1, site], cmvmul(U[4, site], Y[1, siteμ⁺]))
        dU[4, site] = traceless_antihermitian(Bₙ - Cₙ)
    end
end 
