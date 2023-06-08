abstract type AbstractIntegrator end

struct HMCUpdate{TI,TG,TS,TM} <: AbstractUpdate
    steps::Int64
    Δτ::Float64
    P::Liefield
    _temp_U::Gaugefield{TG}
    _temp_staple::Temporaryfield
    _temp_force::Liefield
    _temp_force2::Union{Nothing, Liefield} # second force field for smearing
    _temp_fieldstrength::Union{Nothing, Vector{Temporaryfield}}
    smearing::TS
    metadynamics::TM

    function HMCUpdate(
        U,
        integrator,
        steps,
        Δτ;
        numsmear = 0,
        ρ_stout = 0,
        meta_enabled = false,
    )
        P = Liefield(U)
        _temp_U = similar(U)

        _temp_staple = Temporaryfield(U)
        _temp_force = Liefield(U)

        if meta_enabled
            metadynamics = MetaEnabled()
            _temp_fieldstrength = Vector{Temporaryfield}(undef, 4)

            for i in 1:4
                _temp_fieldstrength[i] = Temporaryfield(U)
            end
        else
            metadynamics = MetaDisabled()
            _temp_fieldstrength = nothing
        end

        smearing = StoutSmearing(U, numsmear, ρ_stout)

        TI = getfield(AbstractUpdateModule, Symbol(integrator))
        TG = eltype(U)
        TS = typeof(smearing)
        TM = typeof(metadynamics)

        if TS == NoSmearing && TM == MetaDisabled
            _temp_force2 = nothing
        else
            _temp_force2 = Liefield(U)
        end


        return new{TI,TG,TS,TM}(
			steps,
			Δτ,
			P,
			_temp_U,
            _temp_staple,
            _temp_force,
            _temp_force2,
            _temp_fieldstrength,
            smearing,
			metadynamics,
		)
    end
end

function update!(
    updatemethod::HMCUpdate{TI,TG,TS,TM},
    U,
    verbose::VerboseLevel;
    Bias = nothing,
    metro_test = true,
) where {TI,TG,TS,TM}
    U_old = updatemethod._temp_U
    substitute_U!(U_old, U)
    gaussian_momenta!(updatemethod.P)
    
    trP2_old = -calc_kinetic_energy(updatemethod.P)

    integrator! = TI()
    integrator!(
        U,
        updatemethod,
        Bias,
    )

    hmc_smearing = updatemethod.smearing

    if TS == NoSmearing
        Sg_old = U.Sg
        Sg_new = calc_gauge_action(U)
    else
        calc_smearedU!(hmc_smearing, U_old)
        fully_smeared_Uold = hmc_smearing.Usmeared_multi[end]
        Sg_old = calc_gauge_action(fully_smeared_Uold)
        calc_smearedU!(hmc_smearing, U)
        fully_smeared_Unew = hmc_smearing.Usmeared_multi[end]
        Sg_new = calc_gauge_action(fully_smeared_Unew)
    end

    trP2_new = -calc_kinetic_energy(updatemethod.P)

    ΔP2 = trP2_new - trP2_old
    ΔSg = Sg_new - Sg_old

    if TM == MetaEnabled
        CV_old = U.CV
        calc_smearedU!(Bias.smearing, U)
        fully_smeared_U = Bias.smearing.Usmeared_multi[end]
        CV_new = top_charge(fully_smeared_U, Bias.kind_of_cv)
        ΔV = Bias(CV_new) - Bias(CV_old)
    else
        CV_old = U.CV
        CV_new = CV_old
        ΔV = 0
    end

    ΔH = ΔP2 + ΔSg + ΔV

    println_verbose2(
        verbose,
        "ΔP2 = ", ΔP2, "\n",
        "ΔS = ", ΔSg, "\n",
        "ΔBias = ", ΔV, "\n",
        "ΔH = ", ΔH,
    )
    accept = metro_test ? rand() ≤ exp(-ΔH) : true

    if accept
        U.Sg = Sg_new
        U.CV = CV_new
        TM == MetaEnabled ? update_bias!(Bias, CV_new) : nothing
        println_verbose2(verbose, "Accepted")
    else
        substitute_U!(U, U_old)
        println_verbose2(verbose, "Rejected")
    end

    return accept
end

include("hmc_integrators.jl")

function updateU!(U, method, fac)
    NX, NY, NZ, NT = size(U)
    ϵ = method.Δτ * fac
    P = method.P

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        @inbounds U[μ][ix,iy,iz,it] = 
                            exp_iQ(-im * ϵ * P[μ][ix,iy,iz,it]) * U[μ][ix,iy,iz,it]
                    end
                end
            end
        end
    end 

    return nothing
end

function updateP!(U, method::HMCUpdate{TI,TG,TS,TM}, fac, Bias) where {TI,TG,TS,TM}
    ϵ = method.Δτ * fac
    P = method.P
    staples = method._temp_staple
    force = method._temp_force
    temp_force = method._temp_force2
    gauge_smearing = method.smearing

    if TM == MetaEnabled
        fieldstrength = method._temp_fieldstrength
        bias_smearing = Bias.smearing
        kind_of_cv = Bias.kind_of_cv
        calc_dQdU_bare!(
            force,
            temp_force,
            fieldstrength,
            U,
            kind_of_cv,
            bias_smearing,
        )
        fully_smeared_U = get_layer(bias_smearing, length(bias_smearing))
        cv = top_charge(fully_smeared_U, kind_of_cv)
        ϵ_bias = ϵ * ∂V∂Q(Bias, cv)
        add!(P, force, ϵ_bias)
    end

    calc_dSdU_bare!(force, temp_force, staples, U, gauge_smearing)
    add!(P, force, ϵ)
    return nothing
end

function calc_dSdU_bare!(
    dSdU,
    temp_force,
    staples,
    U,
    smearing,
)
    if typeof(smearing) == NoSmearing
        calc_dSdU!(dSdU, staples, U)
    else
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        calc_dSdU!(dSdU, staples, fully_smeared_U)
        stout_backprop!(
            dSdU,
            temp_force,
            smearing,
        )
    end

    return nothing
end

function calc_dSdU!(
    dSdU,
    staples,
    U::Gaugefield{T},
) where {T}
    NX, NY, NZ, NT = size(U)
    β = U.β
    staple = T()

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    @inbounds for μ in 1:4
                        A = staple(U, μ, site)
                        staples[μ][site] = A
                        UA = U[μ][site] * A'
                        dSdU[μ][site] = -β/6 * traceless_antihermitian(UA)
                    end

                end
            end
        end
    end

    return nothing
end

function calc_dQdU_bare!(
    dQdU,
    temp_force,
    F,
    U,
    kind_of_charge,
    smearing,
)
    if typeof(smearing) == NoSmearing
        calc_dQdU!(dQdU, F, U, kind_of_charge)
    else
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        calc_dQdU!(dQdU, F, fully_smeared_U, kind_of_charge)
        stout_backprop!(
            dQdU,
            temp_force,
            smearing,
        )
    end

    return nothing
end

function calc_dQdU!(
    dQdU,
    F,
    U,
    kind_of_charge,
)
    NX, NY, NZ, NT = size(U)

    fieldstrength_eachsite!(F, U, kind_of_charge)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    tmp1 = U[1][site] * (
                        ∇trFμνFρσ_clover(U, F, 1, 2, 3, 4, site) -
                        ∇trFμνFρσ_clover(U, F, 1, 3, 2, 4, site) +
                        ∇trFμνFρσ_clover(U, F, 1, 4, 2, 3, site)
                    )
                    dQdU[1][site] = 1/4π^2 * traceless_antihermitian(tmp1)
                    
                    tmp2 = U[2][site] * (
                        ∇trFμνFρσ_clover(U, F, 2, 3, 1, 4, site) -
                        ∇trFμνFρσ_clover(U, F, 2, 1, 3, 4, site) -
                        ∇trFμνFρσ_clover(U, F, 2, 4, 1, 3, site)
                    )
                    dQdU[2][site] = 1/4π^2 * traceless_antihermitian(tmp2)

                    tmp3 = U[3][site] * (
                        ∇trFμνFρσ_clover(U, F, 3, 1, 2, 4, site) -
                        ∇trFμνFρσ_clover(U, F, 3, 2, 1, 4, site) +
                        ∇trFμνFρσ_clover(U, F, 3, 4, 1, 2, site)
                    )
                    dQdU[3][site] = 1/4π^2 * traceless_antihermitian(tmp3)

                    tmp4 = U[4][site] * (
                        ∇trFμνFρσ_clover(U, F, 4, 2, 1, 3, site) -
                        ∇trFμνFρσ_clover(U, F, 4, 1, 2, 3, site) -
                        ∇trFμνFρσ_clover(U, F, 4, 3, 1, 2, site)
                    )
                    dQdU[4][site] = 1/4π^2 * traceless_antihermitian(tmp4)
                end
            end
        end
    end

    return nothing
end

"""
Derivative of the F_μν ⋅ F_ρσ term for Field strength tensor given by plaquette
"""
function ∇trFμνFρσ_plaq(
    U,
    F,
    μ,
    ν,
    ρ,
    σ,
    site::SiteCoords,
)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    component = 
        U[ν][siteμp] * U[μ][siteνp]' * U[ν][site]' * F[ρ][σ][site] +
        U[ν][siteμpνn]' * U[μ][siteνn]' * F[ρ][σ][siteνn] * U[ν][siteνn]

    return im/2 * component
end

"""
Derivative of the F_μν ⋅ F_ρσ term for Field strength tensor given by 1x1-Clover
"""
function ∇trFμνFρσ_clover(
    U,
    F,
    μ,
    ν,
    ρ,
    σ,
    site::SiteCoords,
)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνp = move(siteμp, ν, 1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    component = 
        U[ν][siteμp]    * U[μ][siteνp]'     * U[ν][site]'     * F[ρ][σ][site] +
        U[ν][siteμp]    * U[μ][siteνp]'     * F[ρ][σ][siteνp] * U[ν][site]'   +
        U[ν][siteμp]    * F[ρ][σ][siteμpνp] * U[μ][siteνp]'   * U[ν][site]'   +
        F[ρ][σ][siteμp] * U[ν][siteμp]      * U[μ][siteνp]'   * U[ν][site]'   -
        U[ν][siteμpνn]' * U[μ][siteνn]'     * U[ν][siteνn]    * F[ρ][σ][site] -
        U[ν][siteμpνn]' * U[μ][siteνn]'     * F[ρ][σ][siteνn] * U[ν][siteνn]  -
        U[ν][siteμpνn]' * F[ρ][σ][siteμpνn] * U[μ][siteνn]'   * U[ν][siteνn]  -
        F[ρ][σ][siteμp] * U[ν][siteμpνn]'   * U[μ][siteνn]'   * U[ν][siteνn]

    return im/8 * component
end