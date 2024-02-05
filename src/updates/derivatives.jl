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
    @assert size(dU) == size(staples) == size(U)
    β = U.β

    @threads for site in eachindex(U)
        for μ in 1:4
            A = staple(U, μ, site)
            staples[μ,site] = A
            UA = cmatmul_od(U[μ,site], A)
            dU[μ,site] = -β/6 * traceless_antihermitian(UA)
        end
    end

    return nothing
end

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
    @assert size(dU) == size(F) == size(U)
    c = fac / 4π^2

    @threads for site in eachindex(U)
        tmp1 = cmatmul_oo(U[1,site], (∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
                                      ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)))
        dU[1,site] = c * traceless_antihermitian(tmp1)

        tmp2 = cmatmul_oo(U[2,site], (∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)))
        dU[2,site] = c * traceless_antihermitian(tmp2)

        tmp3 = cmatmul_oo(U[3,site], (∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
                                      ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)))
        dU[3,site] = c * traceless_antihermitian(tmp3)

        tmp4 = cmatmul_oo(U[4,site], (∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)))
        dU[4,site] = c * traceless_antihermitian(tmp4)
    end

    return nothing
end

"""
Derivative of the FμνFρσ term for Field strength tensor given by plaquette
"""
function ∇trFμνFρσ(::Plaquette, U, F, μ, ν, ρ, σ, site)
    Nμ = size(U)[1+μ]
    Nν = size(U)[1+ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    component =
        cmatmul_oddo(U[ν,siteμp]  , U[μ,siteνp], U[ν,site]    , F[ρ,σ,site]) +
        cmatmul_ddoo(U[ν,siteμpνn], U[μ,siteνn], F[ρ,σ,siteνn], U[ν,siteνn])

    return im/2 * component
end

"""
Derivative of the FμνFρσ term for Field strength tensor given by 1x1-Clover
"""
function ∇trFμνFρσ(::Clover, U, F, μ, ν, ρ, σ, site)
    Nμ = size(U)[1+μ]
    Nν = size(U)[1+ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνp = move(siteμp, ν, 1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    # get reused matrices up to cache (can precalculate some products too)
    # Uνsiteμ⁺ = U[ν,siteμp]
    # Uμsiteν⁺ = U[μ,siteνp]
    # Uνsite = U[ν,site]
    # Uνsiteμ⁺ν⁻ = U[ν,siteμpνn]
    # Uμsiteν⁻ = U[μ,siteνn]
    # Uνsiteν⁻ = U[ν,siteνn]

    component =
        cmatmul_oddo(U[ν,siteμp]  , U[μ,siteνp]    , U[ν,site]    , F[ρ,σ,site]) +
        cmatmul_odod(U[ν,siteμp]  , U[μ,siteνp]    , F[ρ,σ,siteνp], U[ν,site])   +
        cmatmul_oodd(U[ν,siteμp]  , F[ρ,σ,siteμpνp], U[μ,siteνp]  , U[ν,site])   +
        cmatmul_oodd(F[ρ,σ,siteμp], U[ν,siteμp]    , U[μ,siteνp]  , U[ν,site])   -
        cmatmul_ddoo(U[ν,siteμpνn], U[μ,siteνn]    , U[ν,siteνn]  , F[ρ,σ,site]) -
        cmatmul_ddoo(U[ν,siteμpνn], U[μ,siteνn]    , F[ρ,σ,siteνn], U[ν,siteνn]) -
        cmatmul_dodo(U[ν,siteμpνn], F[ρ,σ,siteμpνn], U[μ,siteνn]  , U[ν,siteνn]) -
        cmatmul_oddo(F[ρ,σ,siteμp], U[ν,siteμpνn]  , U[μ,siteνn]  , U[ν,siteνn])

    return im/8 * component
end
