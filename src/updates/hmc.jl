import ..Gaugefields: SymanzikTadGaugeAction
abstract type AbstractIntegrator end

struct HMC{TI,TG,TT,TS,PO,F2,FS,IO} <: AbstractUpdate
    steps::Int64
    Δτ::Float64
    friction::Float64

    P::TT
    P_old::PO # second momentum field for GHMC
    U_old::TG
    staples::TT
    force::TT
    force2::F2 # second force field for smearing
    fieldstrength::FS # fieldstrength fields for Bias
    smearing::TS

    fp::IO

    function HMC(U, integrator, trajectory, steps, friction = 0, numsmear = 0, ρ_stout = 0,
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

        P = Temporaryfield(U)
        TT = typeof(P)
        gaussian_TA!(P, 0)
        P_old = friction==0 ? nothing : Temporaryfield(U)
        PO = typeof(P_old)
        U_old = similar(U)
        staples = Temporaryfield(U)
        force = Temporaryfield(U)

        smearing = StoutSmearing(U, numsmear, ρ_stout)
        TS = typeof(smearing)
        force2 = (TS==NoSmearing && !bias_enabled) ? nothing : Temporaryfield(U)
        F2 = typeof(force2)

        if TS == NoSmearing
            @level1("|  ACTION SMEARING: Disabled")
        else
            @level1("|  ACTION SMEARING: $(numsmear) x $(ρ_stout) Stout")
        end

        if bias_enabled
            @level1("|  Bias enabled")
            fieldstrength = Tensorfield(U)
        else
            @level1("|  Bias disabled")
            fieldstrength = nothing
        end
        FS = typeof(fieldstrength)

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
        return new{TI,TG,TT,TS,PO,F2,FS,typeof(fp)}(
            steps, Δτ, friction,
            P, P_old, U_old, staples, force, force2, fieldstrength, smearing,
            fp,
        )
    end
end

include("hmc_integrators.jl")

function update!(hmc::HMC{TI}, U;
    bias::T=nothing, metro_test=true, friction=hmc.friction) where {TI,T}
    U_old = hmc.U_old
    P_old = hmc.P_old
    substitute_U!(U_old, U)
    gaussian_TA!(hmc.P, friction)
    P_old≢nothing && substitute_U!(P_old, hmc.P)

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

        if P_old ≢ nothing # flip momenta if rejected
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
    @assert dims(U) == dims(P)

    @batch for site in eachindex(U)
        for μ in 1:4
            U[μ,site] = cmatmul_oo(exp_iQ(-im*ϵ*P[μ,site]), U[μ,site])
        end
    end

    return nothing
end

function updateP!(U, hmc::HMC, fac, bias::T) where {T}
    ϵ = hmc.Δτ * fac
    P = hmc.P
    @assert dims(U) == dims(P)
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

function Base.close(hmc::HMC)
    hmc.fp isa IOStream && close(hmc.fp)
    return nothing
end
