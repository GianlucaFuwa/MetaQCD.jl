import ..Gaugefields: SymanzikTadGaugeAction
abstract type AbstractIntegrator end

struct HMC{TI,TG,TS,PO,F2,FS,IO} <: AbstractUpdate
    steps::Int64
    Δτ::Float64
    friction::Float64

    P::Liefield
    P_old::PO # second momentum field for GHMC
    U_old::TG
    staples::Temporaryfield
    force::Temporaryfield
    force2::F2 # second force field for smearing
    fieldstrength::FS # fieldstrength fields for Bias
    smearing::TS

    fp::IO

    function HMC(U, integrator, trajectory, steps, friction = π/2, numsmear = 0, ρ_stout = 0,
                 verboselevel = 1; bias_enabled = false, logdir = "")
        @level1("┌ Setting HMC...")
        integrator = Unicode.normalize(integrator, casefold=true)

        if integrator == "leapfrog"
            TI = Leapfrog
        elseif integrator == "omf2slow"
            TI = OMF2Slow
        elseif integrator == "omf2"
            TI = OMF2
        elseif integrator == "omf4slow"
            TI = OMF4Slow
        elseif integrator == "omf4"
            TI = OMF4
        else
            error("HMC integrator \"$(integrator)\" not supported")
        end

        @level1("|  INTEGRATOR: $(TI)")
        TG = typeof(U)
        Δτ = trajectory / steps

        @level1("|  TRAJECTORY LENGTH: $(trajectory)")
        @level1("|  STEPS: $(steps)")
        @level1("|  STEP LENGTH: $(Δτ)")
        @level1("|  FRICTION: $(friction) $(ifelse(friction==π/2, "(default)", ""))")

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
            @level1("|  ACTION SMEARING: Disabled")
        else
            @level1("|  ACTION SMEARING: $(numsmear) x $(ρ_stout) Stout")
        end

        if bias_enabled
            @level1("|  Bias enabled")
            fieldstrength = Vector{Temporaryfield}(undef, 4)

            for i in 1:4
                fieldstrength[i] = Temporaryfield(U)
            end
        else
            @level1("|  Bias disabled")
            fieldstrength = nothing
        end

        if verboselevel>=2 && logdir!=""
            hmc_log_file = logdir * "/hmc_acc_logs.txt"
            @level1("|  Acceptance data tracked in $(hmc_log_file)")
            fp = open(hmc_log_file, "w")
            str = @sprintf("%-22s\t%-22s\t%-22s\t%-22s", "ΔSg", "ΔP²", "ΔV", "ΔH")
            println(fp, str)
        else
            fp = nothing
        end

        @level1("└\n")
        return new{TI,TG,TS,typeof(P_old),typeof(force2),typeof(fieldstrength),typeof(fp)}(
            steps, Δτ, friction,
            P, P_old, U_old, staples, force, force2, fieldstrength, smearing,
            fp,
        )
    end
end

include("hmc_integrators.jl")

function update!(hmc::HMC{TI,TG,TS,PO,F2,FS,IO}, U;
                 bias::T=nothing, metro_test=true) where {TI,TG,TS,PO,F2,FS,IO,T}
    U_old = hmc.U_old
    P_old = hmc.P_old
    substitute_U!(U_old, U)
    gaussian_momenta!(hmc.P, hmc.friction)
    PO≢Nothing && substitute_U!(P_old, hmc.P)

    Sg_old = U.Sg
    trP²_old = -calc_kinetic_energy(hmc.P)

    evolve!(TI(), U, hmc, bias)

    Sg_new = calc_gauge_action(hmc.smearing, U)
    trP²_new = -calc_kinetic_energy(hmc.P)

    ΔP² = trP²_new - trP²_old
    ΔSg = Sg_new - Sg_old

    if T ≡ Nothing
        CV_old = U.CV
        CV_new = CV_old
        ΔV = 0
    else
        CV_old = calc_CV(U_old, bias)
        CV_new = calc_CV(U, bias)
        ΔV = bias(CV_new) - bias(CV_old)
    end

    ΔH = ΔP² + ΔSg + ΔV

    print_hmc_data(hmc.fp, ΔSg, ΔP², ΔV, ΔH)

    accept = metro_test ? rand()≤exp(-ΔH) : true

    if accept
        U.Sg = Sg_new
        U.CV = CV_new
        @level2("|    Accepted")
    else
        substitute_U!(U, U_old)

        if PO ≢ Nothing # flip momenta if rejected
            substitute_U!(hmc.P, P_old)
            mul!(hmc.P, -1)
        end
        @level2("|    Rejected")
    end
    normalize!(U)
    return accept
end

function updateU!(U, hmc, fac)
    ϵ = hmc.Δτ * fac
    P = hmc.P

    @batch per=thread for site in eachindex(U)
        for μ in 1:4
            U[μ][site] = cmatmul_oo(exp_iQ(-im*ϵ*P[μ][site]), U[μ][site])
        end
    end

    return nothing
end

function updateP!(U, hmc::HMC, fac, bias::T) where {T}
    ϵ = hmc.Δτ * fac
    P = hmc.P
    staples = hmc.staples
    force = hmc.force
    temp_force = hmc.force2
    smearing = hmc.smearing
    fieldstrength = hmc.fieldstrength

    if T ≢ Nothing
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

    @batch per=thread for site in eachindex(U)
        tmp1 = cmatmul_oo(U[1][site], (∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
                                       ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
                                       ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)))
        dVdU[1][site] = bias_derivative * 1/4π^2 * traceless_antihermitian(tmp1)

        tmp2 = cmatmul_oo(U[2][site], (∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
                                       ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
                                       ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)))
        dVdU[2][site] = bias_derivative * 1/4π^2 * traceless_antihermitian(tmp2)

        tmp3 = cmatmul_oo(U[3][site], (∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
                                       ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
                                       ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)))
        dVdU[3][site] = bias_derivative * 1/4π^2 * traceless_antihermitian(tmp3)

        tmp4 = cmatmul_oo(U[4][site], (∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
                                       ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
                                       ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)))
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

    component = cmatmul_oddo(U[ν][siteμp]  , U[μ][siteνp], U[ν][site]     , F[ρ][σ][site]) +
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

function print_hmc_data(fp::T, ΔSg, ΔP², ΔV, ΔH) where {T}
    T≡Nothing && return nothing
    str = @sprintf("%+22.15E\t%+22.15E\t%+22.15E\t%+22.15E", ΔSg, ΔP², ΔV, ΔH)
    println(fp, str)
    flush(fp)
    return nothing
end

function Base.close(hmc::HMC{TI,TG,TS,PO,F2,FS,IO}) where {TI,TG,TS,PO,F2,FS,IO}
    IO===IOStream && close(hmc.fp)
    return nothing
end
