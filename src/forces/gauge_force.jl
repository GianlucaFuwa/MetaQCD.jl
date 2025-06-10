"""
    calc_dSdU_bare!(
        dU::Colorfield, staples::Colorfield, U::Gaugefield, temp_force, smearing
    )

Calculate the derivative of the gauge action with respect to the `Gaugefield` `U` and store
the result in `dU`. `staples` and `temp_force` are `Colorfield`s used to store
intermediate results. `smearing` is a `NoSmearing` or `StoutSmearing` object.
"""
calc_dSdU_bare!(dU, staples, U, ::Any, ::NoSmearing) = calc_dSdU!(dU, staples, U)

function calc_dSdU_bare!(dU, staples, U, temp_force, smearing)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    calc_dSdU!(dU, staples, fully_smeared_U)
    stout_backprop!(dU, temp_force, smearing)
    return nothing
end

function calc_dSdU!(
    dU::Colorfield{B,T}, staples::Colorfield{B,T}, U::Gaugefield{B,T},
) where {B,T}
    gauge_action_deriv!(dU, staples, U)
    return nothing
end
