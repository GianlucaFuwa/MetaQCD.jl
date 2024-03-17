"""
Calculate the gauge force for a molecular dynamics step, i.e. the derivative of the
gauge action w.r.t. the bare/unsmeared field U. \\
Needs the additional field "temp_force" to be a TemporaryField when doing recursion
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
    βover6m = float_type(U)(-U.β / 6)
    GA = gauge_action(U)()

    @batch for site in eachindex(U)
        for μ in 1:4
            A = staple(GA, U, μ, site)
            staples[μ, site] = A
            UA = cmatmul_od(U[μ, site], A)
            dU[μ, site] = βover6m * traceless_antihermitian(UA)
        end
    end

    return nothing
end
