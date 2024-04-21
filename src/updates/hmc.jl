import ..Gaugefields: SymanzikTadGaugeAction
abstract type AbstractIntegrator end

"""
    HMC(U, integrator, trajectory, steps, friction = 0, numsmear = 0, ρ_stout = 0,
        verboselevel = 1; bias_enabled = false, logdir = "")

Create an `HMC` object, that can be used as an update algorithm.

# Arguments
- `U`: The gauge field on which the update is performed.
- `integrator`: The integrator used to evolve the field.
- `trajectory`: The length of the HMC trajectory.
- `steps`: The number of integrator steps within the trajectory.
- `friction`: Friction factor in the GHMC algorithm. Has to be in the range [0, 1].
- `numsmear`: Number of Stout smearing steps applied to the gauge action.
- `ρ_stout`: Step length of the Stout smearing applied to the gauge action.
- `verboselevel`: If `2` or higher, creates a logfile in `logdir` containing information
on the trajectories
- `bias_enabled`: If `true`, additional fields are initialized that are needed for Stout
force recursion when using a bias.

# Supported Integrators
- `Leapfrog`
- `OMF2`
- `OMF2Slow`
- `OMF4`
- `OMF4Slow`

# Returns
An HMC object. If `bias_enabled` is `true` then a bias can be passed when updating
a `Gaugefield` with this object. If not, then the required tensors and so on are not
initialized during construction.
"""
struct HMC{TI,TG,TT,TF,TS,PO,F2,FS,IO} <: AbstractUpdate
    steps::Int64
    Δτ::Float64
    friction::Float64

    P::TT
    P_old::PO # second momentum field for GHMC
    U_old::TG
    χ::TF # normally distributed pseudofermions 
    ϕ::TF # Dχ
    staples::TT
    force::TT
    force2::F2 # second force field for smearing
    fieldstrength::FS # fieldstrength fields for Bias
    smearing::TS

    fp::IO
    function HMC(
        U,
        integrator,
        trajectory,
        steps,
        friction=0,
        numsmear=0,
        ρ_stout=0,
        verboselevel=1;
        fermions="none",
        bias_enabled=false,
        logdir="",
    )
        @level1("┌ Setting HMC...")
        integrator = Unicode.normalize(integrator; casefold=true)

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
        P_old = friction == 0 ? nothing : Temporaryfield(U)
        PO = typeof(P_old)
        U_old = Gaugefield(U)
        staples = Temporaryfield(U)
        force = Temporaryfield(U)

        smearing = StoutSmearing(U, numsmear, ρ_stout)
        TS = typeof(smearing)
        force2 = (TS == NoSmearing && !bias_enabled) ? nothing : Temporaryfield(U)
        F2 = typeof(force2)

        if TS == NoSmearing
            @level1("|  ACTION SMEARING: Disabled")
        else
            @level1("|  ACTION SMEARING: $(numsmear) x $(ρ_stout) Stout")
        end

        if fermions == "staggered"
            @level1("|  Dynamical Staggered fermions enabled")
            χ = Fermionfield(U; staggered=true)
            ψ = Fermionfield(χ)
        elseif fermions == "wilson"
            @level1("|  Dynamical Wilson fermions enabled")
            χ = Fermionfield(U; staggered=false)
            ψ = Fermionfield(χ)
        elseif fermions == "none" || fermions === nothing
            @level1("|  Dynamical fermions disabled")
            χ = nothing
            ψ = nothing
        else
            throw(AssertionError("Dynamical fermions \"$fermions\" not supported"))
        end
        TF = typeof(χ)

        if bias_enabled
            @level1("|  Bias enabled")
            fieldstrength = Tensorfield(U)
        else
            @level1("|  Bias disabled")
            fieldstrength = nothing
        end
        FS = typeof(fieldstrength)

        if verboselevel >= 2 && logdir != ""
            hmc_log_file = logdir * "/hmc_acc_logs.txt"
            @level1("|  Acceptance data tracked in $(hmc_log_file)")
            fp = open(hmc_log_file, "w")
            str = @sprintf(
                "%-22s\t%-22s\t%-22s\t%-22s\t%-22s", "ΔP²", "ΔSg", "ΔSf", "ΔV", "ΔH"
            )
            println(fp, str)
        else
            fp = nothing
        end

        @level1("└\n")
        return new{TI,TG,TT,TF,TS,PO,F2,FS,typeof(fp)}(
            steps,
            Δτ,
            friction,
            P,
            P_old,
            U_old,
            χ,
            ϕ,
            staples,
            force,
            force2,
            fieldstrength,
            smearing,
            fp,
        )
    end
end

include("hmc_integrators.jl")

function update!(
    hmc::HMC{TI},
    U;
    fermion_action::TF=nothing,
    bias::TB=nothing,
    metro_test=true,
    friction=hmc.friction,
) where {TI,TF,TB}
    U_old = hmc.U_old
    P_old = hmc.P_old
    P = hmc.P
    ϕ = hmc.ϕ
    smearing = hmc.smearing

    copy!(U_old, U)
    gaussian_TA!(P, friction)
    gaussian_pseudofermions!(ϕ)
    P_old ≢ nothing && copy!(P_old, P)

    trP²_old = -calc_kinetic_energy(P)
    Sg_old = U.Sg

    evolve!(TI(), U, hmc, bias)

    trP²_new = -calc_kinetic_energy(P)
    Sg_new = calc_gauge_action(smearing, U)

    ΔP² = trP²_new - trP²_old
    ΔSg = Sg_new - Sg_old

    if TF ≡ Nothing
        Sf_old = U.Sf
        Sf_new = Sf_old
        ΔSf = 0
    else
        Sf_old = calc_fermion_action(smearing, fermion_action, U_old, ϕ)
        Sf_new = calc_fermion_action(smearing, fermion_action, U, ϕ)
        ΔSf = Sf_new - Sf_old
    end

    if TB ≡ Nothing
        CV_old = U.CV
        CV_new = CV_old
        ΔV = 0
    else
        CV_old = calc_CV(U_old, bias)
        CV_new = calc_CV(U, bias)
        ΔV = bias(CV_new) - bias(CV_old)
    end

    ΔH = ΔP² + ΔSg + ΔV + ΔSf

    print_hmc_data(hmc.fp, ΔP², ΔSg, ΔSf, ΔV, ΔH)

    accept = metro_test ? rand() ≤ exp(-ΔH) : true

    if accept
        U.Sg = Sg_new
        U.Sf = Sf_new
        U.CV = CV_new
        @level2("|    Accepted")
    else
        copy!(U, U_old)

        if P_old ≢ nothing # flip momenta if rejected
            copy!(P, P_old)
            mul!(P, -1)
        end
        @level2("|    Rejected")
    end
    normalize!(U)
    return accept
end

function updateU!(U::Gaugefield{CPU,T}, hmc, fac) where {T}
    ϵ = T(hmc.Δτ * fac)
    P = hmc.P
    @assert dims(U) == dims(P)

    @batch for site in eachindex(U)
        for μ in 1:4
            U[μ, site] = cmatmul_oo(exp_iQ(-im * ϵ * P[μ, site]), U[μ, site])
        end
    end

    return nothing
end

function updateP!(U, hmc::HMC, fac, fermion_action::TF, bias::TB) where {TB,TF}
    ϵ = hmc.Δτ * fac
    P = hmc.P
    @assert dims(U) == dims(P)
    staples = hmc.staples
    force = hmc.force
    temp_force = hmc.force2
    smearing = hmc.smearing
    fieldstrength = hmc.fieldstrength

    if TB ≢ Nothing
        calc_dVdU_bare!(force, fieldstrength, U, temp_force, bias)
        add!(P, force, ϵ)
    end

    if TF ≢ Nothing
        calc_dSfdU_bare!(force, fermion_action, U, hmc.ϕ, temp_force, smearing)
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

calc_fermion_action(::NoSmearing, fermion_action, U, ψ) = calc_fermion_action(U)

function calc_fermion_action(smearing::StoutSmearing, fermion_action, U, ψ)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    smeared_action = calc_fermion_action(fermion_action, fully_smeared_U, ψ)
    return smeared_action
end

print_hmc_data(::Nothing, args...) = nothing

function print_hmc_data(fp::T, ΔP², ΔSg, ΔSf, ΔV, ΔH) where {T}
    T ≡ Nothing && return nothing
    str = @sprintf("%+22.15E\t%+22.15E\t%+22.15E\t%+22.15E", ΔP², ΔSg, ΔSf, ΔV, ΔH)
    println(fp, str)
    flush(fp)
    return nothing
end

function Base.close(hmc::HMC)
    hmc.fp isa IOStream && close(hmc.fp)
    return nothing
end
