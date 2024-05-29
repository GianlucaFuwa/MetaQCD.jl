abstract type AbstractIntegrator end

"""
    HMC(
        U,
        integrator,
        trajectory,
        steps,
        friction = 0,
        numsmear = 0,
        ρ_stout = 0;
        hmc_logging = true,
        fermion_action = nothing,
        heavy_flavours = 0,
        bias_enabled = false,
        logdir = "",
    )

Create an `HMC` object, that can be used as an update algorithm.

# Arguments
- `U`: The gauge field on which the update is performed.
- `integrator`: The integrator used to evolve the field.
- `trajectory`: The length of the HMC trajectory.
- `steps`: The number of integrator steps within the trajectory.
- `friction`: Friction factor in the GHMC algorithm. Has to be in the range [0, 1].
- `numsmear`: Number of Stout smearing steps applied to the gauge action.
- `ρ_stout`: Step length of the Stout smearing applied to the gauge action.
- `hmc_logging`: If true, creates a logfile in `logdir` containing information
on the trajectories, unless `logdir = ""`
- `fermion_action`: An `AbstratFermionAction` to initialize the appropriate fermion fields
- `heavy_flavours`: The number of non-degenerate heavy flavours, again to initialize the
right number of fermion fields
- `bias_enabled`: If `true`, additional fields are initialized that are needed for Stout
force recursion when using a bias.

# Supported Integrators
- `Leapfrog`
- `OMF2`
- `OMF2Slow`
- `OMF4`
- `OMF4Slow`

# Supported Fermion Actions
- `WilsonFermionAction`
- `StaggeredFermionAction`
- `StaggeredEOPreFermionAction`

# Returns
An HMC object, which can be used as an argument in the `update!` function.
"""
struct HMC{TI,TG,TT,TF,TSG,TSF,PO,F2,FS,TIO} <: AbstractUpdate
    steps::Int64
    Δτ::Float64
    friction::Float64

    P::TT
    P_old::PO # second momentum field for GHMC
    U_old::TG
    ϕ::TF
    staples::TT
    force::TT
    force2::F2 # second force field for smearing
    fieldstrength::FS # fieldstrength fields for Bias
    smearing_gauge::TSG
    smearing_fermion::TSF

    fp::TIO
    function HMC(
        U,
        integrator::Type{<:AbstractIntegrator},
        trajectory,
        steps,
        friction=0,
        numsmear_gauge=0,
        numsmear_fermion=0,
        ρ_stout_gauge=0,
        ρ_stout_fermion=0;
        hmc_logging=true,
        fermion_action=nothing,
        heavy_flavours=0,
        bias_enabled=false,
        logdir="",
    )
        @level1("┌ Setting HMC...")
        @level1("|  INTEGRATOR: $(integrator)")
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

        smearing_gauge = StoutSmearing(U, numsmear_gauge, ρ_stout_gauge)
        smearing_fermion = StoutSmearing(U, numsmear_fermion, ρ_stout_fermion)
        TSG = typeof(smearing_gauge)
        TSF = typeof(smearing_fermion)
        has_smearing = TSG != NoSmearing || TSF != NoSmearing
        force2 = (!has_smearing && !bias_enabled) ? nothing : Temporaryfield(U)
        F2 = typeof(force2)

        if TSG == NoSmearing
            @level1("|  Gauge Action Smearing disabled")
        else
            @level1("|  Gauge Action Smearing: $(numsmear_gauge) x $(ρ_stout_gauge) Stout")
        end

        if TSF == NoSmearing
            @level1("|  Fermion Action Smearing disabled")
        else
            @level1(
                "|  Fermion Action Smearing: $(numsmear_fermion) x $(ρ_stout_fermion) Stout"
            )
        end

        if fermion_action === StaggeredFermionAction
            @level1("|  Dynamical Staggered fermions enabled")
            ϕ = ntuple(_ -> Fermionfield(U; staggered=true), 1 + heavy_flavours)
        elseif fermion_action === StaggeredEOPreFermionAction
            @level1("|  Dynamical EO-Pre Staggered fermions enabled")
            ϕ = ntuple(_ -> even_odd(Fermionfield(U; staggered=true)), 1 + heavy_flavours)
        elseif fermion_action === WilsonFermionAction
            @level1("|  Dynamical Wilson fermions enabled")
            ϕ = ntuple(_ -> Fermionfield(U), 1 + heavy_flavours)
        elseif fermion_action == "none" || fermion_action === nothing
            @level1("|  Dynamical fermions disabled")
            ϕ = nothing
        else
            throw(AssertionError("Dynamical fermions \"$fermion_action\" not supported"))
        end
        TF = typeof(ϕ)

        if bias_enabled
            @level1("|  Bias enabled")
            fieldstrength = Tensorfield(U)
        else
            @level1("|  Bias disabled")
            fieldstrength = nothing
        end
        FS = typeof(fieldstrength)

        if hmc_logging && logdir != ""
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
        return new{integrator,TG,TT,TF,TSG,TSF,PO,F2,FS,typeof(fp)}(
            steps,
            Δτ,
            friction,
            P,
            P_old,
            U_old,
            ϕ,
            staples,
            force,
            force2,
            fieldstrength,
            smearing_gauge,
            smearing_fermion,
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
    if TF !== Nothing
        @assert TF <: Tuple "fermion_action must be nothing or a tuple of fermion actions"
        @assert hmc.ϕ !== nothing "fermion_action passed but not activated in HMC"
    end

    U_old = hmc.U_old
    P_old = hmc.P_old
    P = hmc.P
    ϕ = hmc.ϕ
    smearing_gauge = hmc.smearing_gauge
    smearing_fermion = hmc.smearing_fermion

    copy!(U_old, U)
    gaussian_TA!(P, friction)
    if TF !== Nothing
        for i in eachindex(fermion_action)
            sample_pseudofermions!(ϕ[i], fermion_action[i], U, smearing_fermion, i)
        end
    end
    P_old ≢ nothing && copy!(P_old, P)

    trP²_old = -calc_kinetic_energy(P)
    Sg_old = U.Sg

    evolve!(TI(), U, hmc, fermion_action, bias)

    trP²_new = -calc_kinetic_energy(P)
    Sg_new = calc_gauge_action(smearing_gauge, U)

    ΔP² = trP²_new - trP²_old
    ΔSg = Sg_new - Sg_old

    if TF ≡ Nothing
        Sf_old = 0.0
        Sf_new = 0.0
        ΔSf = 0
    else
        Sf_old = 0.0
        Sf_new = 0.0
        for i in eachindex(fermion_action)
            Sf_old += calc_fermion_action(smearing_fermion, fermion_action[i], U_old, ϕ[i])
            Sf_new += calc_fermion_action(smearing_fermion, fermion_action[i], U, ϕ[i])
        end
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
    check_dims(U, P)

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
    staples = hmc.staples
    force = hmc.force
    ϕ = hmc.ϕ
    temp_force = hmc.force2
    smearing_gauge = hmc.smearing_gauge
    smearing_fermion = hmc.smearing_fermion
    fieldstrength = hmc.fieldstrength

    if TB ≢ Nothing
        calc_dVdU_bare!(force, fieldstrength, U, temp_force, bias)
        add!(P, force, ϵ)
    end

    if TF ≢ Nothing
        for i in eachindex(fermion_action)
            calc_dSfdU_bare!(
                force, fermion_action[i], U, ϕ[i], temp_force, smearing_fermion
            )
            add!(P, force, ϵ)
        end
    end

    calc_dSdU_bare!(force, staples, U, temp_force, smearing_gauge)
    add!(P, force, ϵ)
    return nothing
end

calc_gauge_action(::NoSmearing, U) = calc_gauge_action(U)

function calc_gauge_action(smearing::StoutSmearing, U)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    smeared_gauge_action = calc_gauge_action(fully_smeared_U)
    return smeared_gauge_action
end

function calc_fermion_action(::NoSmearing, fermion_action, U, ϕ)
    return calc_fermion_action(fermion_action, U, ϕ)
end

function calc_fermion_action(smearing::StoutSmearing, fermion_action, U, ϕ)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    smeared_fermion_action = calc_fermion_action(fermion_action, fully_smeared_U, ϕ)
    return smeared_fermion_action
end

function sample_pseudofermions!(ϕ, fermion_action, U, ::NoSmearing, ::Any)
    sample_pseudofermions!(ϕ, fermion_action, U)
    return nothing
end

function sample_pseudofermions!(ϕ, fermion_action, U, smearing::StoutSmearing, i)
    # we only need to smear once even if we have multiple fermion actions
    i == 1 && calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    sample_pseudofermions!(ϕ, fermion_action, fully_smeared_U)
    return nothing
end

print_hmc_data(::Nothing, args...) = nothing

function print_hmc_data(fp::T, ΔP², ΔSg, ΔSf, ΔV, ΔH) where {T}
    T ≡ Nothing && return nothing
    str = @sprintf(
        "%+22.15E\t%+22.15E\t%+22.15E\t%+22.15E\t%+22.15E", ΔP², ΔSg, ΔSf, ΔV, ΔH
    )
    println(fp, str)
    flush(fp)
    return nothing
end

function Base.close(hmc::HMC)
    hmc.fp isa IOStream && close(hmc.fp)
    return nothing
end
