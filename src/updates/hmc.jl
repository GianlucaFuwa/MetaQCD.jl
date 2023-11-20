import ..Gaugefields: SymanzikTadGaugeAction
abstract type AbstractIntegrator end

mutable struct HMC{TI,TG,TS} <: AbstractUpdate
    trajectory::Float64
    steps::Int64
    Δτ::Float64
    friction::Float64

    direction::Int64
    U₀::Float64
    P₀::Float64
    P::Liefield
    P_old::Union{Nothing, Liefield} # second momentum field for GHMC
    U_old::TG
    staples::Temporaryfield
    force::Temporaryfield
    force2::Union{Nothing, Temporaryfield} # second force field for smearing
    fieldstrength::Union{Nothing, Vector{Temporaryfield}} # fieldstrength fields for Bias
    smearing::TS

    fp::Union{Nothing, IOStream}

    function HMC(
        U,
        integrator, steps, trajectory;
        verbose = nothing,
        friction = π/2,
        numsmear = 0,
        ρ_stout = 0,
        bias_enabled = false,
        verboselevel = 1,
        logdir = "",
    )
        println_verbose1(verbose, ">> Setting HMC...")
        TI = getfield(Updates, Symbol(integrator))
        println_verbose1(verbose, "\t>> INTEGRATOR = $(TI)")
        TG = typeof(U)
        Δτ = trajectory / steps

        println_verbose1(verbose, "\t>> TRAJECTORY LENGTH = $(trajectory)")
        println_verbose1(verbose, "\t>> STEPS = $(steps)")
        println_verbose1(verbose, "\t>> STEP LENGTH = $(Δτ)")
        println_verbose1(verbose, "\t>> FRICTION = $(friction)")

        P = Liefield(U)
        gaussian_momenta!(P, π/2)
        P_old = friction==π/2 ? nothing : Liefield(U)
        U_old = similar(U)
        staples = Temporaryfield(U)
        force = Temporaryfield(U)

        smearing = StoutSmearing(U, numsmear, ρ_stout)
        TS = typeof(smearing)
        force2 = (TS==NoSmearing && !bias_enabled) ? nothing : Temporaryfield(U)

        if TS == NoSmearing
            println_verbose1(verbose, "\t>> NO ACTION SMEARING")
        else
            println_verbose1(verbose, "\t>> ACTION SMEARING = (ρ = $(ρ_stout), n = $(numsmear))")
        end

        if bias_enabled
            println_verbose1(verbose, "\t>> BIAS ENABLED")
            fieldstrength = Vector{Temporaryfield}(undef, 4)

            for i in 1:4
                fieldstrength[i] = Temporaryfield(U)
            end
        else
            println_verbose1(verbose, "\t>> BIAS DISABLED")
            fieldstrength = nothing
        end

        if verboselevel>=2 && logdir!=""
            hmc_log_file = logdir * "/hmc_acc_logs.txt"
            println_verbose1(verbose, "\t>> ACCEPTANCE DATA TRACKED IN $(hmc_log_file)")
            fp = open(hmc_log_file, "w")
            @printf(
                fp, "%-22s\t%-22s\t%-22s\t%-22s\t%-22s\n", "ΔSg", "ΔP²", "ΔV", "P₀", "ΔH"
            )
        else
            fp = nothing
        end

        println_verbose1(verbose, "")

        return new{TI,TG,TS}(
			trajectory, steps, Δτ, friction,
            1, 0.0, 0.0, P, P_old, U_old, staples, force, force2, fieldstrength, smearing,
            fp,
		)
    end
end

include("hmc_integrators.jl")

function update!(hmc::HMC{TI,TG,TS}, U, verbose; bias=nothing, metro_test=true) where {TI,TG,TS}
    U_old = hmc.U_old
    P_old = hmc.P_old
    substitute_U!(U_old, U)
    gaussian_momenta!(hmc.P, hmc.friction)
    P_old≢nothing && substitute_U!(P_old, hmc.P)
    hmc.direction = rand()<0.5 ? 1 : -1

    U₀_old = hmc.U₀
    Sg_old = U.Sg
    trP²_old = -calc_kinetic_energy(hmc.P)

    evolve!(TI(), U, hmc, bias)

    Sg_new = calc_gauge_action(hmc.smearing, U)
    trP²_new = -calc_kinetic_energy(hmc.P)

    ΔP² = trP²_new - trP²_old
    ΔSg = Sg_new - Sg_old

    if bias ≡ nothing
        CV_old = U.CV
        CV_new = CV_old
        ΔV = 0
    else
        CV_old = calc_CV(U_old, bias)
        CV_new = calc_CV(U, bias)
        ΔV = bias(CV_new, hmc.U₀) - bias(CV_old, U₀_old)
    end

    ΔH = ΔP² + ΔSg + ΔV + hmc.P₀

    print_hmc_data(hmc.fp, ΔSg, ΔP², ΔV, ΔH, hmc.P₀)

    accept = metro_test ? rand()≤exp(-ΔH) : true

    if accept
        U.Sg = Sg_new
        U.CV = CV_new
        println_verbose2(verbose, "Accepted")
    else
        substitute_U!(U, U_old)

        if P_old ≢ nothing# flip momenta if rejected
            substitute_U!(hmc.P, P_old)
            mul!(hmc.P, -1)
        end
        println_verbose2(verbose, "Rejected")
    end

    normalize!(U)
    return accept
end

function updateU!(U, hmc, fac)
    ϵ = hmc.Δτ * fac
    P = hmc.P
    hmc.U₀ += hmc.direction*ϵ / hmc.trajectory

    @batch per=thread for site in eachindex(U)
        for μ in 1:4
            U[μ][site] = cmatmul_oo(exp_iQ(-im*ϵ*P[μ][site]), U[μ][site])
        end
    end

    return nothing
end

function updateP!(U, hmc::HMC, fac, bias)
    ϵ = hmc.Δτ * fac
    P = hmc.P
    staples = hmc.staples
    force = hmc.force
    temp_force = hmc.force2
    smearing = hmc.smearing
    fieldstrength = hmc.fieldstrength

    if bias ≢ nothing
        cv = calc_dVdU_bare!(force, fieldstrength, U, temp_force, bias, hmc.U₀)
        hmc.P₀ -= ∂V∂t(bias, cv, hmc.U₀) * ϵ * hmc.direction / hmc.trajectory
        @printf("t = %.10f\tcv = %.10f\tP₀ = %.10f\n", hmc.U₀, cv, hmc.P₀)
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

    @batch per=thread for site in eachindex(U)
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
function calc_dVdU_bare!(dVdU, F, U, temp_force, bias, t)
    smearing = bias.smearing
    cv = calc_CV(U, bias)
    bias_derivative = ∂V∂Q(bias, cv, t)

    if typeof(smearing) == NoSmearing
        calc_dVdU!(kind_of_cv(bias), dVdU, F, U, bias_derivative)
    else
        fully_smeared_U = smearing.Usmeared_multi[end]
        calc_dVdU!(kind_of_cv(bias), dVdU, F, fully_smeared_U, bias_derivative)
        stout_backprop!(dVdU, temp_force, smearing)
    end
    return cv
end

function calc_dVdU!(kind_of_charge, dVdU, F, U, bias_derivative)
    fieldstrength_eachsite!(kind_of_charge, F, U)

    @batch per=thread for site in eachindex(U)
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

calc_gauge_action(::NoSmearing, U) = calc_gauge_action(U)

function calc_gauge_action(smearing::StoutSmearing, U)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    smeared_action = calc_gauge_action(fully_smeared_U)
    return smeared_action
end

print_hmc_data(::Nothing, args...) = nothing

function print_hmc_data(fp, ΔSg, ΔP², ΔV, ΔH, P₀)
    @printf(fp, "%+22.15E\t%+22.15E\t%+22.15E\t%+22.15E\t%+22.15E\n", ΔSg, ΔP², ΔV, P₀, ΔH)
    flush(fp)
    return nothing
end
