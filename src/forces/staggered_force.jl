"""
    calc_dSfdU!(dU, U, ψ, fermion_action)

Solve the equation `Dϕ = ψ` for `ϕ`, where `D` is a Dirac operator
and store the result in `ϕ`. The `temps` argument is a list of temporary fields that
are used to store intermediate results.
"""
function calc_dSfdU!(dU, U, ψ, fermion_action::StaggeredFermionAction)
    @assert dims(dU) == dims(U) == dims(ψ)
    temps = get_temps(fermion_action)
    D = fermion_action.D(U)
    # (D†D)⁻¹ψ
    temp = fermion_action.temp_F
    DdagD = Hermitian(D)
    return solve_D⁻¹x!(temp, DdagD, ψ, temps...)
end
