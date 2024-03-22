const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

function calc_dSfdU!( 
    dU,
    U,
    ψ::WilsonFermionfield,
    fermion_action::WilsonFermionAction,
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

