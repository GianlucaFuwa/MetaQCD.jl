"""
    calc_dVdU_bare!(dU::Colorfield, F::Tensorfield, U, temp_force, bias, is_smeared)

Calculate the bias force for a molecular dynamics step, i.e. the derivative of the
bias potential w.r.t. the bare/unsmeared field U.

If `temp_force isa Colorfield` and `bias.smearing != nothing`, the derivative is calculated
w.r.t. the fully smeared field V = 𝔉(U) using Stout smearing and Stout force recursion.

If `is_smeared = true`, it is assumed that smearing has already been applied to `U`,
meaning that the gauge fields in `bias.smearing` are the smeared versions of `U`
"""
function calc_dVdU_bare!(dU, F, U, temp_force, bias, icv, is_smeared)
    cv = calc_cv(U, bias, icv, is_smeared)
    bias_derivative = ∂V∂Q(bias, cv, icv)
    smearing = bias.smearing
    calc_cv_deriv_bare!(dU, bias, F, U, temp_force, smearing, icv, bias_derivative)
    return nothing
end

function calc_cv_deriv_bare!(dU, bias, F, U, ::Any, ::NoSmearing, icv, fac=1)
    itemp = _unwrap_val(bias.bias[icv].cvinfo.cv_temp_ind)
    calc_cv_deriv!(dU, bias, icv, F[itemp], U, fac)
    return nothing
end

function calc_cv_deriv_bare!(
    dU, bias, F, ::Gaugefield, temp_force, smearing::StoutSmearing, icv, fac=1
)
    # We don't need to smear here, because the calc_cv() call from above has already
    # done it, meaning that bias.smearing.Usmeared_mulit[end] is the fully smeared field
    # that we need
    level = bias.cv_numsmears[icv]
    smeared_U = smearing.Usmeared_multi[level+1]
    itemp = _unwrap_val(bias.bias[icv].cvinfo.cv_temp_ind)
    calc_cv_deriv!(dU, bias, icv, F[itemp], smeared_U, fac)
    stout_backprop!(dU, temp_force, smearing, level)
    return nothing
end
