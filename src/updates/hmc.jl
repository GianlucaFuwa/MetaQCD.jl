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
"""
struct HMC{TI,TG,TT,TF,TSG,TSF,PO,F2,FS,TFP1,TFP2} <: AbstractUpdate
    integrator::TI
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

    logfile::TFP1
    forcefile::TFP2
    function HMC(
        integrator, steps, Δτ, friction, P, P_old, U_old, ϕ, staples, force, force2,
        fieldstrength, smearing_gauge, smearing_fermion, logfile, forcefile,
    )
        @level1("┌ Setting HMC...")
        @level1("|  INTEGRATOR: $(string(integrator))")
        @level1("|  TRAJECTORY LENGTH: $(steps * Δτ)")
        @level1("|  STEPS: $(steps)")
        @level1("|  STEP LENGTH: $(Δτ)")
        @level1("|  FRICTION: $(friction) $(ifelse(friction==0, "(default)", ""))")
        isnothing(fieldstrength) ? @level1("|  BIAS DISABLED") : @level1("|  BIAS ENABLED")
        @level1("|  GAUGE SMEARING: $(string(smearing_gauge))")
        @level1("|  FERMION SMEARING: $(string(smearing_fermion))")
        !isnothing(logfile) && @level1("|  HMC LOGFILE: $(logfile)")
        !isnothing(forcefile) && @level1("|  HMC FORCEFILE: $(forcefile)")
        @level1("└\n")
        TI = typeof(integrator)
        TG = typeof(U_old)
        TT = typeof(staples)
        TF = typeof(ϕ)
        TSG = typeof(smearing_gauge)
        TSF = typeof(smearing_fermion)
        PO = typeof(P_old)
        F2 = typeof(force2)
        FS = typeof(fieldstrength)
        TFP1 = typeof(logfile)
        TFP2 = typeof(forcefile)
        return new{TI,TG,TT,TF,TSG,TSF,PO,F2,FS,TFP1,TFP2}(
            integrator, steps, Δτ, friction, P, P_old, U_old, ϕ, staples, force, force2,
            fieldstrength, smearing_gauge, smearing_fermion, logfile, forcefile,
        )
    end
end

function HMC(
    U,
    integrator::AbstractIntegrator,
    trajectory,
    steps,
    friction=0.0,
    numsmear_gauge=0,
    numsmear_fermion=0,
    ρ_stout_gauge=0.0,
    ρ_stout_fermion=0.0;
    hmc_logging=true,
    fermion_action=nothing,
    heavy_flavours=0,
    bias_enabled=false,
    logdir="",
)
    Δτ = trajectory/steps
    P = Colorfield(U)
    gaussian_TA!(P, 0)
    P_old = friction == 0 ? nothing : Colorfield(U)
    U_old = Gaugefield(U)
    staples = Colorfield(U)
    force = Colorfield(U)

    smearing_gauge = StoutSmearing(U, numsmear_gauge, ρ_stout_gauge)
    smearing_fermion = if isnothing(fermion_action)
        NoSmearing()
    else
        StoutSmearing(U, numsmear_fermion, ρ_stout_fermion)
    end

    has_smearing = smearing_gauge != NoSmearing() || smearing_fermion != NoSmearing()
    force2 = (!has_smearing && !bias_enabled) ? nothing : Colorfield(U)

    if fermion_action === StaggeredFermionAction
        ϕ = ntuple(_ -> Fermionfield(U; staggered=true), 1 + heavy_flavours)
    elseif fermion_action === StaggeredEOPreFermionAction
        ϕ = ntuple(_ -> even_odd(Fermionfield(U; staggered=true)), 1 + heavy_flavours)
    elseif fermion_action === WilsonFermionAction
        ϕ = ntuple(_ -> Fermionfield(U), 1 + heavy_flavours)
    elseif fermion_action == "none" || fermion_action === nothing
        ϕ = nothing
    else
        throw(AssertionError("Dynamical fermions \"$fermion_action\" not supported"))
    end

    fieldstrength = bias_enabled ? Tensorfield(U) : nothing

    if hmc_logging && logdir != ""
        # XXX: Probably want swap this too in MPI PT-MetaD
        logfile = joinpath(logdir, "hmc_acc_logs.txt")
        open(logfile, "w") do fp
            @printf(
                fp,
                "%-22s\t%-22s\t%-22s\t%-22s\t%-22s\n",
                "ΔP²", "ΔSg", "ΔSf", "ΔV", "ΔH",
            )
        end

        if !isnothing(ϕ)
            forcefile = joinpath(logdir, "hmc_force_logs.txt")
            force_fp = fopen(forcefile, "w")
            printf(force_fp, "%-25s", "|F_Sg|")
            for i in eachindex(ϕ)
                printf(force_fp, "%-25s", "|F_Sf$i|")
            end
            if bias_enabled
                printf(force_fp, "%-25s", "|F_V|")
            end
            printf(force_fp, "\n")
            fclose(force_fp)
        else
            forcefile = nothing
        end
    else
        logfile = nothing
        forcefile = nothing
    end

    return HMC(
        integrator, steps, Δτ, friction, P, P_old, U_old, ϕ, staples, force, force2,
        fieldstrength, smearing_gauge, smearing_fermion, logfile, forcefile,
    )
end

include("hmc_integrators.jl")

function update!(
    hmc::HMC{TI},
    U;
    fermion_action::TF=nothing,
    bias::TB=NoBias(),
    metro_test::Bool=true,
    therm::Bool=false,
) where {TI,TF,TB}
    if TF !== Nothing
        @assert TF <: Tuple "fermion_action must be nothing or a tuple of fermion actions"
        @assert !isnothing(hmc.ϕ) "fermion_action passed but not activated in HMC"
    end

    U_old = hmc.U_old
    P_old = hmc.P_old
    P = hmc.P
    ϕ = hmc.ϕ
    smearing_gauge = hmc.smearing_gauge
    # Check if bias and fermion smearing have same parameters
    shared_smearing = (bias == NoBias()) ? false : (bias.smearing == hmc.smearing_fermion)
    smearing_fermion = shared_smearing ? bias.smearing : hmc.smearing_fermion
    friction = therm ? 0.0 : hmc.friction

    copy!(U_old, U)
    gaussian_TA!(P, friction)
    !isnothing(P_old) && copy!(P_old, P)

    trP²_old = -calc_kinetic_energy(P)
    Sg_old = calc_gauge_action(U, smearing_gauge)
    CV_old = calc_CV(U, bias)
    V_old = bias(CV_old)
    sample_pseudofermions!(ϕ, fermion_action, U, smearing_fermion, shared_smearing)
    Sf_old = calc_fermion_action(fermion_action, U, ϕ, smearing_fermion, true)

    evolve!(hmc.integrator, U, hmc, fermion_action, bias)

    trP²_new = -calc_kinetic_energy(P)
    Sg_new = calc_gauge_action(U, smearing_gauge)
    CV_new = calc_CV(U, bias)
    V_new = bias(CV_new)
    Sf_new = calc_fermion_action(fermion_action, U, ϕ, smearing_fermion, shared_smearing)

    ΔP² = trP²_new - trP²_old
    ΔSg = Sg_new - Sg_old
    ΔV = V_new - V_old
    ΔSf = Sf_new - Sf_old

    ΔH = ΔP² + ΔSg + ΔV + ΔSf
    accept = metro_test ? rand() ≤ exp(-ΔH) : true
    print_hmc_data(hmc.logfile, ΔP², ΔSg, ΔSf, ΔV, ΔH)

    if accept
        U.Sg = Sg_new
        U.CV = CV_new
        @level2("|    Accepted")
    else
        copy!(U, U_old)

        if !isnothing(P_old) # flip momenta if rejected
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

function updateP!(U, hmc::HMC, fac, fermion_action, bias)
    ϵ = hmc.Δτ * fac
    P = hmc.P
    staples = hmc.staples
    force = hmc.force
    ϕ = hmc.ϕ
    temp_force = hmc.force2
    smearing_gauge = hmc.smearing_gauge
    shared_smearing = (bias == NoBias()) ? false : (bias.smearing == hmc.smearing_fermion)
    smearing_fermion = shared_smearing ? bias.smearing : hmc.smearing_fermion
    fieldstrength = hmc.fieldstrength

    fp = !isnothing(hmc.forcefile) ? fopen(hmc.forcefile, "a") : nothing

    calc_dSdU_bare!(force, staples, U, temp_force, smearing_gauge)
    if !isnothing(fp)
        fnorm = norm(force)
        printf(fp, "%+-25.15E", fnorm)
    end
    add!(P, force, ϵ)

    if !isnothing(fermion_action)
        for i in eachindex(fermion_action)
            calc_dSfdU_bare!(
                force, fermion_action[i], U, ϕ[i], temp_force, smearing_fermion, i>1
            )
            if !isnothing(fp)
                fnorm = norm(force)
                printf(fp, "%+-25.15E", fnorm)
            end
            add!(P, force, ϵ)
        end
    end

    if bias isa Bias
        calc_dVdU_bare!(force, fieldstrength, U, temp_force, bias, shared_smearing)
        if !isnothing(fp)
            fnorm = norm(force)
            printf(fp, "%+-25.15E", fnorm)
        end
        add!(P, force, ϵ)
    end

    if !isnothing(fp)
        printf(fp, "\n")
        fclose(fp)
    end
    return nothing
end

calc_gauge_action(U, ::NoSmearing) = calc_gauge_action(U)

function calc_gauge_action(U, smearing::StoutSmearing)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    smeared_gauge_action = calc_gauge_action(fully_smeared_U)
    return smeared_gauge_action
end

sample_pseudofermions!(::Any, ::Nothing, ::Any, ::NoSmearing, ::Bool) = nothing

function sample_pseudofermions!(ϕ, fermion_action, U, ::NoSmearing, ::Any)
    for i in eachindex(fermion_action)
        sample_pseudofermions!(ϕ[i], fermion_action[i], U)
    end
    return nothing
end

function sample_pseudofermions!(ϕ, fermion_action, U, smearing::StoutSmearing, is_smeared)
    # we only need to smear once even if we have multiple fermion actions
    is_smeared || calc_smearedU!(smearing, U)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    for i in eachindex(fermion_action)
        sample_pseudofermions!(ϕ[i], fermion_action[i], fully_smeared_U)
    end
    return nothing
end

calc_fermion_action(::Nothing, ::Any, ::Any, ::NoSmearing, ::Any) = 0.0

function calc_fermion_action(fermion_action, U, ϕ, ::NoSmearing, ::Any)
    Sf = 0.0
    for i in eachindex(fermion_action)
        Sf += calc_fermion_action(fermion_action[i], U, ϕ[i])
    end
    return Sf
end

function calc_fermion_action(fermion_action, U, ϕ, smearing::StoutSmearing, is_smeared)
    is_smeared || calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    Sf = 0.0
    for i in eachindex(fermion_action)
        Sf += calc_fermion_action(fermion_action[i], fully_smeared_U, ϕ[i])
    end
    return Sf
end

@inline print_hmc_data(::Nothing, args...) = nothing

@inline function print_hmc_data(logfile, ΔP², ΔSg, ΔSf, ΔV, ΔH)
    fp = fopen(logfile, "a")
    printf(fp, "%+22.15E\t", ΔP²)
    printf(fp, "%+22.15E\t", ΔSg)
    printf(fp, "%+22.15E\t", ΔSf)
    printf(fp, "%+22.15E\t", ΔV)
    printf(fp, "%+22.15E\n", ΔH)
    fclose(fp)
    return nothing
end

# In order to write and load the bias easily with JLD2 for checkpointing, we need to define
# custom serialization, because saving and loading IOStreams doesn't work
using JLD2

struct HMCSerialization{TI,TG,TT,TF,TSG,TSF,PO,F2,FS,TFP1,TFP2}
    integrator::TI
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

    logfile::TFP1
    forcefile::TFP2
end

function JLD2.writeas(
    ::Type{<:HMC{TI,TG,TT,TF,TSG,TSF,PO,F2,FS,TFP1,TFP2}}
) where {TI,TG,TT,TF,TSG,TSF,PO,F2,FS,TFP1,TFP2}
    return HMCSerialization{TI,TG,TT,TF,TSG,TSF,PO,F2,FS,TFP1,TFP2}
end

function Base.convert(::Type{<:HMCSerialization}, hmc::HMC)
    out = HMCSerialization(
        hmc.integrator,
        hmc.steps,
        hmc.Δτ,
        hmc.friction,
        hmc.P,
        hmc.P_old,
        hmc.U_old,
        hmc.ϕ,
        hmc.staples,
        hmc.force,
        hmc.force2,
        hmc.fieldstrength,
        hmc.smearing_gauge,
        hmc.smearing_fermion,
        hmc.logfile,
        hmc.forcefile,
    )
    return out
end

function Base.convert(::Type{<:HMC}, hmc::HMCSerialization)
    out = HMC(
        hmc.integrator,
        hmc.steps,
        hmc.Δτ,
        hmc.friction,
        hmc.P,
        hmc.P_old,
        hmc.U_old,
        hmc.ϕ,
        hmc.staples,
        hmc.force,
        hmc.force2,
        hmc.fieldstrength,
        hmc.smearing_gauge,
        hmc.smearing_fermion,
        hmc.logfile,
        hmc.forcefile,
    )
    return out
end
