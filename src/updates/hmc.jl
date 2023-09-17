import ..Gaugefields: SymanzikTadGaugeAction
abstract type AbstractIntegrator end

struct HMCUpdate{TI,TG,TS,TB} <: AbstractUpdate
    steps::Int64
    Δτ::Float64
    ϕ::Float64
    P::Liefield
    _temp_U::TG
    _temp_staple::Temporaryfield
    _temp_force::Temporaryfield
    _temp_force2::Union{Nothing, Temporaryfield} # second force field for smearing
    _temp_fieldstrength::Union{Nothing, Vector{Temporaryfield}}
    smearing::TS
    fp::Union{Nothing, IOStream}

    function HMCUpdate(
        U,
        integrator,
        steps,
        Δτ,
        ϕ;
        numsmear = 0,
        ρ_stout = 0,
        bias_enabled = false,
        verboselevel = 1,
        logdir = "",
    )
        GA = eltype(U)
        @assert GA != SymanzikTadGaugeAction "Tadpole improved actions not supported in HMC"
        P = Liefield(U)
        gaussian_momenta!(P, π/2)
        _temp_U = similar(U)

        _temp_staple = Temporaryfield(U)
        _temp_force = Temporaryfield(U)

        if bias_enabled
            TB = BiasEnabled
            _temp_fieldstrength = Vector{Temporaryfield}(undef, 4)

            for i in 1:4
                _temp_fieldstrength[i] = Temporaryfield(U)
            end
        else
            TB = BiasDisabled
            _temp_fieldstrength = nothing
        end

        smearing = StoutSmearing(U, numsmear, ρ_stout)

        TI = getfield(Updates, Symbol(integrator))
        TG = typeof(U)
        TS = typeof(smearing)

        if TS == NoSmearing && !bias_enabled
            _temp_force2 = nothing
        else
            _temp_force2 = Temporaryfield(U)
        end

        if verboselevel >= 2
            fp = open(logdir * "/hmc_$(GA)_$(TI)_logs.txt", "w")
            println(
                fp,
                rpad("ΔSg", 22, " "), "\t",
                rpad("ΔP2", 22, " "), "\t",
                rpad("ΔV", 22, " "), "\t",
                rpad("ΔH", 22, " "),
            )
        else
            fp = nothing
        end

        return new{TI,TG,TS,TB}(
			steps,
			Δτ,
            ϕ,
			P,
			_temp_U,
            _temp_staple,
            _temp_force,
            _temp_force2,
            _temp_fieldstrength,
            smearing,
            fp,
		)
    end
end

include("hmc_integrators.jl")

function update!(
    updatemethod::HMCUpdate{TI,TG,TS,TB},
    U,
    verbose::VerboseLevel;
    bias = nothing,
    metro_test = true,
) where {TI,TG,TS,TB}
    U_old = updatemethod._temp_U
    substitute_U!(U_old, U)
    gaussian_momenta!(updatemethod.P, updatemethod.ϕ)

    trP2_old = -calc_kinetic_energy(updatemethod.P)

    evolve!(TI(), U, updatemethod, bias)

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

    if bias === nothing
        CV_old = U.CV
        CV_new = CV_old
        ΔV = 0
    else
        CV_old = calc_CV(U_old, bias)
        CV_new = calc_CV(U, bias)
        ΔV = bias(CV_new) - bias(CV_old)
    end

    ΔH = ΔP2 + ΔSg + ΔV

    if updatemethod.fp !== nothing
        println(
            updatemethod.fp,
            rpad(@sprintf("%.15E", ΔSg), 22, " "), "\t",
            rpad(@sprintf("%.15E", ΔP2), 22, " "), "\t",
            rpad(@sprintf("%.15E", ΔV), 22, " "), "\t",
            rpad(@sprintf("%.15E", ΔH), 22, " "),
        )
        flush(updatemethod.fp)
    end

    accept = metro_test ? rand() ≤ exp(-ΔH) : true

    if accept
        U.Sg = Sg_new
        U.CV = CV_new
        println_verbose2(verbose, "Accepted")
    else
        substitute_U!(U, U_old)
        updatemethod.ϕ!=π/2 && mul!(updatemethod.P, -1) # flip momenta if rejected
        println_verbose2(verbose, "Rejected")
    end

    normalize!(U)
    return accept
end

function updateU!(U, method, fac)
    ϵ = method.Δτ * fac
    P = method.P

    @batch for site in eachindex(U)
        for μ in 1:4
            U[μ][site] = cmatmul_oo(exp_iQ(-im * ϵ * P[μ][site]), U[μ][site])
        end
    end

    return nothing
end

function updateP!(U, method::HMCUpdate, fac, bias)
    ϵ = method.Δτ * fac
    P = method.P
    staples = method._temp_staple
    force = method._temp_force
    temp_force = method._temp_force2
    smearing = method.smearing

    if bias !== nothing
        fieldstrength = method._temp_fieldstrength
        calc_dVdU_bare!(force, fieldstrength, U, temp_force, bias)
        add!(P, force, ϵ)
    end

    calc_dSdU_bare!(force, staples, U, temp_force, smearing)
    add!(P, force, ϵ)
    return nothing
end

"""
Calculate the gauge force for a molecular dynamics step, i.e. the derivative of the
gauge action w.r.t. the bare/unsmeared field U. \\
Needs the additional field "temp_force" to be a TemporaryField when doing recursion
"""
calc_dSdU_bare!(dSdU, staples, U, ::Any, ::NoSmearing) = calc_dSdU!(dSdU, staples, U)

function calc_dSdU_bare!(dSdU, staples, U, temp_force, smearing)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    calc_dSdU!(dSdU, staples, fully_smeared_U)
    stout_backprop!(dSdU, temp_force, smearing)
    return nothing
end

function calc_dSdU!(dSdU, staples, U)
    β = U.β

    @batch for site in eachindex(U)
        for μ in 1:4
            A = staple(U, μ, site)
            staples[μ][site] = A
            UA = cmatmul_od(U[μ][site], A)
            dSdU[μ][site] = -β/6 * traceless_antihermitian(UA)
        end
    end

    return nothing
end

"""
Calculate the bias force for a molecular dynamics step, i.e. the derivative of the
bias potential w.r.t. the bare/unsmeared field U. \\
Needs the additional field "temp_force" to be a TemporaryField when doing recursion
"""
function calc_dVdU_bare!(dVdU, F, U, temp_force, bias)
    smearing = bias.smearing
    cv = calc_CV(U, bias)
    bias_derivative = ∂V∂Q(bias, cv)

    if typeof(smearing) == NoSmearing
        calc_dVdU!(kind_of_cv(bias), dVdU, F, U, bias_derivative)
    else
        fully_smeared_U = smearing.Usmeared_multi[end]
        calc_dVdU!(kind_of_cv(bias), dVdU, F, fully_smeared_U, bias_derivative)
        stout_backprop!(dVdU, temp_force, smearing)
    end
    return nothing
end

function calc_dVdU!(kind_of_charge, dVdU, F, U, bias_derivative)
    fieldstrength_eachsite!(kind_of_charge, F, U)

    @batch for site in eachindex(U)
        tmp1 = cmatmul_oo(U[1][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
            ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)
        ))
        dVdU[1][site] = bias_derivative * 1/4π^2 * traceless_antihermitian(tmp1)

        tmp2 = cmatmul_oo(U[2][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)
        ))
        dVdU[2][site] = bias_derivative * 1/4π^2 * traceless_antihermitian(tmp2)

        tmp3 = cmatmul_oo(U[3][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
            ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)
        ))
        dVdU[3][site] = bias_derivative * 1/4π^2 * traceless_antihermitian(tmp3)

        tmp4 = cmatmul_oo(U[4][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)
        ))
        dVdU[4][site] = bias_derivative * 1/4π^2 * traceless_antihermitian(tmp4)
    end

    return nothing
end

"""
Derivative of the FμνFρσ term for Field strength tensor given by plaquette
"""
function ∇trFμνFρσ(::Plaquette, U, F, μ, ν, ρ, σ, site)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    component =
        cmatmul_oddo(U[ν][siteμp]  , U[μ][siteνp], U[ν][site]     , F[ρ][σ][site]) +
        cmatmul_ddoo(U[ν][siteμpνn], U[μ][siteνn], F[ρ][σ][siteνn], U[ν][siteνn])

    return im/2 * component
end

"""
Derivative of the FμνFρσ term for Field strength tensor given by 1x1-Clover
"""
function ∇trFμνFρσ(::Clover, U, F, μ, ν, ρ, σ, site)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνp = move(siteμp, ν, 1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    component =
        cmatmul_oddo(U[ν][siteμp]   , U[μ][siteνp]     , U[ν][site]     , F[ρ][σ][site]) +
        cmatmul_odod(U[ν][siteμp]   , U[μ][siteνp]     , F[ρ][σ][siteνp], U[ν][site])    +
        cmatmul_oodd(U[ν][siteμp]   , F[ρ][σ][siteμpνp], U[μ][siteνp]   , U[ν][site])    +
        cmatmul_oodd(F[ρ][σ][siteμp], U[ν][siteμp]     , U[μ][siteνp]   , U[ν][site])    -
        cmatmul_ddoo(U[ν][siteμpνn] , U[μ][siteνn]     , U[ν][siteνn]   , F[ρ][σ][site]) -
        cmatmul_ddoo(U[ν][siteμpνn] , U[μ][siteνn]     , F[ρ][σ][siteνn], U[ν][siteνn])  -
        cmatmul_dodo(U[ν][siteμpνn] , F[ρ][σ][siteμpνn], U[μ][siteνn]   , U[ν][siteνn])  -
        cmatmul_oddo(F[ρ][σ][siteμp], U[ν][siteμpνn]   , U[μ][siteνn]   , U[ν][siteνn])

    return im/8 * component
end
