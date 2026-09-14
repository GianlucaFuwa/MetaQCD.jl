"""
    calc_dScdU_bare!(dU::Colorfield, F::Tensorfield, U, temp_force, constraint)

Calculate the constraint force for a molecular dynamics step, i.e. the derivative of the
constraint action w.r.t. the bare/unsmeared field U.

If `temp_force isa Colorfield` and `constraint.smearing != nothing`, the derivative is calculated
w.r.t. the fully smeared field V = 𝔉(U) using Stout smearing and Stout force recursion.

If `is_smeared = true`, it is assumed that smearing has already been applied to `U`,
meaning that the gauge fields in `constraint.smearing` are the smeared versions of `U`
"""
function calc_dScdU_bare!(dU, F, U, temp_force, constraint, is_smeared=false)
    cv = calc_cv(U, constraint, 1, is_smeared)[1]
    σ = constraint.variance
    μ = constraint.value
    constraint_derivative = (cv - μ) / σ^2
    smearing = constraint.smearing
    calc_cv_deriv_bare!(dU, constraint, F, U, temp_force, smearing, 1, constraint_derivative)
    return cv
end

function calc_cv_deriv_bare!(dU, constraint, F, U, ::Any, ::NoSmearing, ::Any, fac=1)
    itemp = _unwrap_val(constraint.info.cv_temp_ind)
    calc_cv_deriv!(dU, constraint, 1, F[itemp], U, fac)
    return nothing
end

function calc_cv_deriv_bare!(
    dU, constraint, F, ::Gaugefield, temp_force, smearing::StoutSmearing, ::Any=1, fac=1
)
    level = constraint.smearing.numlayers
    smeared_U = smearing.Usmeared_multi[level+1]
    itemp = _unwrap_val(constraint.info.cv_temp_ind)
    calc_cv_deriv!(dU, constraint, 1, F[itemp], smeared_U, fac)
    stout_backprop!(dU, temp_force, smearing, level)
    return nothing
end

