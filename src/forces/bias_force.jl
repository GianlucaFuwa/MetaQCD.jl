"""
Calculate the bias force for a molecular dynamics step, i.e. the derivative of the
bias potential w.r.t. the bare/unsmeared field U. \\
Needs the additional field "temp_force" to be a TemporaryField when doing recursion
"""
function calc_dVdU_bare!(dU, F, U, temp_force, bias::Bias)
    smearing = bias.smearing
    cv = calc_CV(U, bias)
    bias_derivative = ∂V∂Q(bias, cv)

    if typeof(smearing) == NoSmearing
        calc_dQdU!(kind_of_cv(bias), dU, F, U, bias_derivative)
    else
        fully_smeared_U = smearing.Usmeared_multi[end]
        calc_dQdU!(kind_of_cv(bias), dU, F, fully_smeared_U, bias_derivative)
        stout_backprop!(dU, temp_force, smearing)
    end
    return nothing
end

function calc_dQdU_bare!(kind_of_cv, dU, F, U, temp_force=nothing, smearing=NoSmearing())
    if typeof(smearing) == NoSmearing
        fieldstrength_eachsite!(kind_of_cv, F, U)
        calc_dQdU!(kind_of_cv, dU, F, U)
    else
        fully_smeared_U = smearing.Usmeared_multi[end]
        fieldstrength_eachsite!(kind_of_cv, F, fully_smeared_U)
        calc_dQdU!(kind_of_cv, dU, F, fully_smeared_U)
        stout_backprop!(dU, temp_force, smearing)
    end
    return nothing
end

function calc_dQdU!(kind_of_charge, dU, F, U, fac=1.0)
    @assert dims(dU) == dims(F) == dims(U)
    c = float_type(U)(fac / 4π^2)

    @batch for site in eachindex(U)
        tmp1 = cmatmul_oo(
            U[1, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
                ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)
            ),
        )
        dU[1, site] = c * traceless_antihermitian(tmp1)
        tmp2 = cmatmul_oo(
            U[2, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)
            ),
        )
        dU[2, site] = c * traceless_antihermitian(tmp2)
        tmp3 = cmatmul_oo(
            U[3, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
                ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)
            ),
        )
        dU[3, site] = c * traceless_antihermitian(tmp3)
        tmp4 = cmatmul_oo(
            U[4, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)
            ),
        )
        dU[4, site] = c * traceless_antihermitian(tmp4)
    end

    return nothing
end

# """
# Derivative of the FμνFρσ term for Field strength tensor given by plaquette
# """
function ∇trFμνFρσ(::Plaquette, U, F, μ, ν, ρ, σ, site)
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺ = move(site, μ, 1i32, Nμ)
    siteν⁺ = move(site, ν, 1i32, Nν)
    siteν⁻ = move(site, ν, -1i32, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1i32, Nν)

    component =
        cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], F[ρ, σ, site]) +
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], F[ρ, σ, siteν⁻], U[ν, siteν⁻])

    return im * 1//2 * component
end

# """
# Derivative of the FμνFρσ term for Field strength tensor given by 1x1-Clover
# """
function ∇trFμνFρσ(::Clover, U, F, μ, ν, ρ, σ, site)
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺ = move(site, μ, 1i32, Nμ)
    siteν⁺ = move(site, ν, 1i32, Nν)
    siteν⁻ = move(site, ν, -1i32, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1i32, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1i32, Nν)

    # get reused matrices up to cache (can precalculate some products too)
    # Uνsiteμ⁺ = U[ν,siteμ⁺]
    # Uμsiteν⁺ = U[μ,siteν⁺]
    # Uνsite = U[ν,site]
    # Uνsiteμ⁺ν⁻ = U[ν,siteμ⁺ν⁻]
    # Uμsiteν⁻ = U[μ,siteν⁻]
    # Uνsiteν⁻ = U[ν,siteν⁻]

    component =
        cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], F[ρ, σ, site]) +
        cmatmul_odod(U[ν, siteμ⁺], U[μ, siteν⁺], F[ρ, σ, siteν⁺], U[ν, site]) +
        cmatmul_oodd(U[ν, siteμ⁺], F[ρ, σ, siteμ⁺ν⁺], U[μ, siteν⁺], U[ν, site]) +
        cmatmul_oodd(F[ρ, σ, siteμ⁺], U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site]) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻], F[ρ, σ, site]) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], F[ρ, σ, siteν⁻], U[ν, siteν⁻]) -
        cmatmul_dodo(U[ν, siteμ⁺ν⁻], F[ρ, σ, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻]) -
        cmatmul_oddo(F[ρ, σ, siteμ⁺], U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻])

    return im * 1//8 * component
end
