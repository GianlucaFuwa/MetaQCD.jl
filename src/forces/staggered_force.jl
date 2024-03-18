const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

"""
    calc_dSfdU!(dU, U, ψ, fermion_action)

Calculate the derivative of `fermion_action` w.r.t. the gauge field `U` on the fermion
background `ψ` and store the result in `dU`.
"""
function calc_dSfdU!( # Unrooted / Nf=4
    dU,
    U,
    ψ::StaggeredFermionfield,
    fermion_action::StaggeredFermionAction{4},
)
    @assert dims(dU) == dims(U) == dims(ψ)
    DdagD⁻¹ψ = fermion_action.temp1
    temps_for_cg = get_cg_temps(fermion_action)
    D = fermion_action.D(U)
    # ∇ⁱSf = ((D†D)⁻¹ψ)† (∂D\dwⁱ D† + D ∂D†\dwⁱ) ((D†D)⁻¹ψ) 
    DdagD = Hermitian(D)
    solve_D⁻¹x!(DdagD⁻¹ψ, DdagD, ψ, temps_for_cg...)
    return mul!(temp2, D, DdagD⁻¹ψ)
end
