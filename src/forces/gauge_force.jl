"""
    calc_dSdU_bare!(dU, staples, U, temp_force, smearing)

Calculate the derivative of the gauge action with respect to the `Gaugefield` `U` and store
the result in `dU`. `staples` and `temp_force` are `TemporaryFields` used to store
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

function calc_dSdU!(dU, staples, U)
    @assert dims(dU) == dims(staples) == dims(U)
    mβover6 = float_type(U)(-U.β / 6)
    GA = gauge_action(U)()

    @batch for site in eachindex(U)
        for μ in 1:4
            A = staple(GA, U, μ, site)
            staples[μ, site] = A
            UA = cmatmul_od(U[μ, site], A)
            dU[μ, site] = mβover6 * traceless_antihermitian(UA)
        end
    end

    return nothing
end
