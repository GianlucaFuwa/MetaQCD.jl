abstract type AbstractIntegrator end

struct HMCUpdate{I, GA, S, M} <: AbstractUpdate
    integrator!::I
    steps::Int64
    Δτ::Float64
    P::Liefield
    _temp_U::Gaugefield{GA}
    _temp_staple::TemporaryField
    _temp_force::TemporaryField
    _temp_force2::Union{Nothing, TemporaryField} # second force field for smearing
    _temp_fieldstrength::Union{Nothing, Vector{TemporaryField}}
    smearing::S
    metadynamics::M

    function HMCUpdate(
        integrator::String,
        steps,
        Δτ,
        U::Gaugefield{T};
        numsmear = 0,
        ρ_stout = 0,
        meta_enabled = false,
    ) where {T <: AbstractGaugeAction}
        # turn integrator String into Symbol
        integrator = getfield(AbstractUpdateModule, Symbol(integrator))()
        P = Liefield(U)
        _temp_U = similar(U)

        _temp_staple = TemporaryField(U)
        _temp_force = TemporaryField(U)

        if meta_enabled
            metadynamics = MetaEnabled()
            _temp_fieldstrength = Vector{TemporaryField}(undef, 4)

            for i in 1:4
                _temp_fieldstrength[i] = TemporaryField(U)
            end
        else
            metadynamics = MetaDisabled()
            _temp_fieldstrength = nothing
        end

        smearing = StoutSmearing(U, numsmear, ρ_stout)

        if typeof(smearing) == NoSmearing
            _temp_force2 = nothing
        else
            _temp_force2 = TemporaryField(U)
        end

        return new{typeof(integrator), T, typeof(smearing), typeof(metadynamics)}(
			integrator, # turn integrator Symbol into AbstractIntegrator/Function
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
    updatemethod::HMCUpdate{I, GA, S, M},
    U::Gaugefield{GA},
    verbose::VerboseLevel;
    Bias = nothing,
    metro_test = true,
) where {I, GA, S, M}
    U_old = updatemethod._temp_U
    substitute_U!(U_old, U)
    gaussian_momenta!(updatemethod.P)
    
    trP2_old = -calc_kinetic_energy(updatemethod.P)

    updatemethod.integrator!(
        U,
        updatemethod,
        Bias,
    )

    hmc_smearing = updatemethod.smearing

    if typeof(hmc_smearing) == NoSmearing
        Sg_old = U.Sg
        Sg_new = calc_gauge_action(U)
    else
        calc_smearedU!(hmc_smearing, U_old)
        fully_smeared_Uold = hmc_smearing.Usmeared_multi[end]
        Sg_old = calc_gauge_action(fully_smeared_Uold)
        calc_smearedU!(hmc_smearing, U)
        fully_smeared_U = hmc_smearing.Usmeared_multi[end]
        Sg_new = calc_gauge_action(fully_smeared_U)
    end

    trP2_new = -calc_kinetic_energy(updatemethod.P)

    ΔP2 = trP2_new - trP2_old
    ΔSg = Sg_new - Sg_old

    if M == MetaEnabled
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
        M == MetaEnabled ? update_bias!(Bias, CV_new) : nothing
        println_verbose2(verbose, "Accepted")
    else
        substitute_U!(U, U_old)
        println_verbose2(verbose, "Rejected")
    end

    return accept
end

include("hmc_integrators.jl")

function updateU!(U::Gaugefield, method::HMCUpdate, fac)
    NX, NY, NZ, NT = size(U)
    ϵ = method.Δτ * fac
    P = method.P

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        U[μ][ix,iy,iz,it] = exp_iQ(-im * ϵ * P[μ][ix,iy,iz,it]) * U[μ][ix,iy,iz,it]
                    end
                end
            end
        end
    end 

    return nothing
end

function updateP!(U::Gaugefield, method::HMCUpdate{I, GA, S, M}, fac, Bias) where {I, GA, S, M}
    ϵ = method.Δτ * fac
    P = method.P
    staples = method._temp_staple
    force = method._temp_force
    temp_force = method._temp_force2
    gauge_smearing = method.smearing

    if M == MetaEnabled
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
    dSdU::TemporaryField,
    temp_force::Union{Nothing, TemporaryField},
    staples::TemporaryField,
    U::Gaugefield,
    smearing,
)
    if smearing === nothing || typeof(smearing) == NoSmearing
        calc_dSdU!(dSdU, staples, U)
    else
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        calc_dSdU!(dSdU, staples, fully_smeared_U)
        stout_backprop!(
            dSdU,
            temp_force,
            # U,
            smearing,
        )
    end

    return nothing
end

function calc_dSdU!(
    dSdU::TemporaryField,
    staples::TemporaryField,
    U::Gaugefield,
)
    NX, NY, NZ, NT = size(U)
    β = U.β

    staple_eachsite!(staples, U)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        tmp = U[μ][ix,iy,iz,it] * staples[μ][ix,iy,iz,it]'
                        dSdU[μ][ix,iy,iz,it] = -β/6 * traceless_antihermitian(tmp)
                    end
                end
            end
        end
    end

    return nothing
end

function calc_dQdU_bare!(
    dQdU::TemporaryField,
    temp_force::Union{Nothing, TemporaryField},
    Fμν::Vector{TemporaryField},
    U::Gaugefield,
    kind_of_charge,
    smearing::AbstractSmearing,
)
    if typeof(smearing) == NoSmearing
        calc_dQdU!(dQdU, Fμν, U, kind_of_charge)

    else
        numlayers = length(smearing)
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        calc_dQdU!(dQdU, Fμν, fully_smeared_U, kind_of_charge)
        stout_backprop!(
            dQdU,
            temp_force,
            # U,
            smearing,
        )
    end

    return nothing
end

function calc_dQdU!(
    dQdU::TemporaryField,
    Fμν::Vector{TemporaryField},
    U::Gaugefield,
    kind_of_charge::String,
)
    NX, NY, NZ, NT = size(U)

    fieldstrength_eachsite!(Fμν, U, kind_of_charge)

    if kind_of_charge == "plaquette"
        ∇trFμνFρσ = ∇trFμνFρσ_plaq
    elseif kind_of_charge == "clover"
        ∇trFμνFρσ = ∇trFμνFρσ_clover
    else
        error("topological force for charge type $(kind_of_charge) not supported")
    end

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    dQdU[1][ix,iy,iz,it] = 1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 1, 2, 3, 4, site) -
                        ∇trFμνFρσ(U, Fμν, 1, 3, 2, 4, site) +
                        ∇trFμνFρσ(U, Fμν, 1, 4, 2, 3, site)
                    )
                        
                    dQdU[2][ix,iy,iz,it] = -1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 2, 1, 3, 4, site) +
                        ∇trFμνFρσ(U, Fμν, 2, 4, 1, 3, site) -
                        ∇trFμνFρσ(U, Fμν, 2, 3, 1, 4, site)
                    )

                    dQdU[3][ix,iy,iz,it] = 1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 3, 4, 1, 2, site) +
                        ∇trFμνFρσ(U, Fμν, 3, 1, 2, 4, site) -
                        ∇trFμνFρσ(U, Fμν, 3, 2, 1, 4, site)
                    )

                    dQdU[4][ix,iy,iz,it] = -1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 4, 3, 1, 2, site) -
                        ∇trFμνFρσ(U, Fμν, 4, 2, 1, 3, site) +
                        ∇trFμνFρσ(U, Fμν, 4, 1, 2, 3, site)
                    )
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
    U::Gaugefield,
    Fμν::Vector{TemporaryField},
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
        U[ν][siteμp] * U[μ][siteνp]' * U[ν][site]' * Fμν[ρ][σ][site] +
        U[ν][siteμpνn]' * U[μ][siteνn]' * Fμν[ρ][σ][siteνn] * U[ν][siteνn] -

    return im/2 * component
end

"""
Derivative of the F_μν ⋅ F_ρσ term for Field strength tensor given by 1x1-Clover
"""
function ∇trFμνFρσ_clover(
    U::Gaugefield,
    Fμν::Vector{TemporaryField},
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
        U[ν][siteμp] * U[μ][siteνp]' * U[ν][site]' * Fμν[ρ][σ][site] +
        U[ν][siteμp] * U[μ][siteνp]' * Fμν[ρ][σ][siteνp] * U[ν][site]' +
        U[ν][siteμp] * Fμν[ρ][σ][siteμpνp] * U[μ][siteνp]' * U[ν][site]' +
        Fμν[ρ][σ][siteμp] * U[ν][siteμp] * U[μ][siteνp]' * U[ν][site]' -
        U[ν][siteμpνn]' * U[μ][siteνn]' * U[ν][siteνn] * Fμν[ρ][σ][site] -
        U[ν][siteμpνn]' * U[μ][siteνn]' * Fμν[ρ][σ][siteνn] * U[ν][siteνn] -
        U[ν][siteμpνn]' * Fμν[ρ][σ][siteμpνn] * U[μ][siteνn]' * U[ν][siteνn] -
        Fμν[ρ][σ][siteμp] * U[ν][siteμpνn]' * U[μ][siteνn]' * U[ν][siteνn]

    return im/8 * component
end