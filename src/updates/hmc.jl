abstract type AbstractIntegrator end

include("hmc_levels.jl")

"""
    HMC(
        U,
        hmc_levels,
        trajectory,
        friction=0.0,
        numsmear_gauge=0,
        numsmear_fermion=0,
        rho_stout_gauge=0.0,
        rho_stout_fermion=0.0;
        rafriction=0.0,
        hmc_logging=true,
        fermion_action="quenched",
        numfermions=0,
        numcv=0,
        logdir="",
        instance=MPI_INSTANCE[],
    )

Create an `HMC` object, that can be used as an update algorithm.

# Arguments
- `U`: The gauge field on which the update is performed.
- `levels`: A vector of `Dict`s that define the levels of the hmc integration scheme.
To see what parameters are needed see the file "./hmc_levels".
- `trajectory`: The length of the HMC trajectory.
- `steps`: The number of integrator steps within the trajectory.
- `friction`: Friction factor in the GHMC algorithm. Has to be in the range [0, 1].
- `numsmear_gauge`: Number of Stout smearing steps applied to the gauge action.
- `numsmear_fermion`: Number of Stout smearing steps applied to the fermion action.
- `rho_stout_gauge`: Step length of the Stout smearing applied to the gauge action.
- `rho_stout_fermion`: Step length of the Stout smearing applied to the fermion action.
- `rafriction`: Friction parameter for the repell-attract HMC.
- `hmc_logging`: If true, creates a logfile in `logdir` containing information
on the trajectories, unless `logdir = ""`
- `fermion_action`: An String that identifies the fermion action type to initialize the appropriate fermion fields
- `numfermions`: The number of non-degenerate heavy flavours, again to initialize the
right number of fermion fields
- `numcv`: If bigger than 0, additional fields are initialized that are needed for Stout
force recursion when using a bias.
- `logdir`: Directory that hmc data should be written into.
- `instance`: Integer identifier of current instance (for parallel tempering and multiple walkers).

# Supported Fermion Actions
- `quenched`
- `staggered`
- `staggered_eo`
- `wilson`
- `wilson_eo`
"""
struct HMC{TL,NL,TG,TGH,TGO,TP,TT,TF,TSG,TSF,TPO,TPOO,TF2,TFS,TLF} <: AbstractUpdate
    levels::TL
    numlevels::Val{NL}
    friction::Float64
    constraint::Union{Nothing,Float64}

    P::TP
    P_old::TPO # second momentum field for GHMC
    P0::TPOO
    U_old::TG
    U_high::TGH
    U0::TGO
    ϕ::TF
    staples::TT
    force::TT
    force2::TF2 # second force field for smearing
    fieldstrength::TFS # fieldstrength fields for Bias
    smearing_gauge::TSG
    smearing_fermion::TSF

    substep_CVs::Vector{Vector{Float64}}
    substep_CVs_single::Vector{Float64}
    substep_counter::Base.RefValue{Int64}

    logfile::TLF
    function HMC(
        levels::TL,
        numlevels,
        friction,
        constraint,
        P::TP,
        P_old::TPO,
        P0::TPOO,
        U_old::TG,
        U_high::TGH,
        U0::TGO,
        ϕ::TF,
        staples::TT,
        force,
        force2::TF2,
        fieldstrength::TFS,
        smearing_gauge::TSG,
        smearing_fermion::TSF,
        substep_CVs,
        logfile::TLF,
    ) where {TL,TP,TPO,TPOO,TG,TGH,TGO,TF,TT,TF2,TFS,TSG,TSF,TLF}
        @level1("- Constructing HMC...")
        @level1("|  LEVELS:")
        for lvl in reverse(levels)
            @level1("$(string(lvl))")
        end
        @level1("|  FRICTION: $(friction) $(ifelse(friction==0, "(default)", ""))")
        @level1("|  CONSTRAINT: $(constraint)")
        isnothing(fieldstrength) ? @level1("|  BIAS DISABLED") : @level1("|  BIAS ENABLED")
        @level1("|  GAUGE SMEARING: $(string(smearing_gauge))")
        @level1("|  FERMION SMEARING: $(string(smearing_fermion))")
        !isnothing(logfile) && @level1("|  HMC LOGFILE: $(logfile)")
        @level1("-\n")
        substep_CVs_single = isempty(substep_CVs) ? Float64[] : zeros(length(substep_CVs))
        substep_counter = Base.RefValue{Int64}(0)
        NL = _unwrap_val(numlevels)
        return new{TL,NL,TG,TGH,TGO,TP,TT,TF,TSG,TSF,TPO,TPOO,TF2,TFS,TLF}(
            levels,
            numlevels,
            friction,
            constraint,
            P,
            P_old,
            P0,
            U_old,
            U_high,
            U0,
            ϕ,
            staples,
            force,
            force2,
            fieldstrength,
            smearing_gauge,
            smearing_fermion,
            substep_CVs,
            substep_CVs_single,
            substep_counter,
            logfile,
        )
    end
end

function HMC(
    U::Gaugefield{B,T},
    hmc_levels,
    trajectory,
    friction=0.0,
    numsmear_gauge=0,
    numsmear_fermion=0,
    rho_stout_gauge=0.0,
    rho_stout_fermion=0.0;
    velocity=0.1,
    rafriction=0.0,
    constraint=nothing,
    hmc_logging=true,
    fermion_action="quenched",
    numfermions=0,
    numcv=0,
    logdir="",
    instance=MPI_INSTANCE[],
) where {B,T}
    P = Colorfield(U, Float64; no_halo=true)
    gaussian_TA!(P, 0)
    P_old = friction == 0 ? nothing : Colorfield(U; no_halo=true)
    U_old = Gaugefield(U; no_halo=true)
    U_high = T==Float64 ? nothing : Gaugefield(U, Float64)
    U0, P0 = if contains(lowercase(hmc_levels[1]["integrator"]), "constrained")
        Gaugefield(U), Colorfield(U, Float64; no_halo=true)
    else
        nothing, nothing
    end
    staples = Colorfield(U; no_halo=true)
    force = Colorfield(U; no_halo=true)

    numlevels = Val(length(hmc_levels))
    level_params = level_parameters_from_dict(hmc_levels)

    levels = ntuple(numlevels) do i
        lvl = level_params[i]

        if contains(lowercase(lvl.integrator), "constrained")
            @assert length(hmc_levels) == 1 "Constrained HMC can only be used with a single time scale"
        end

        forces = lvl.forces
        numchildren = Val(i - 1)
        Δτ = if i == length(level_params)
            trajectory / lvl.numsteps
        else
            Nᵢ = lvl.numsteps

            for j in i+1:length(hmc_levels)
                Nᵢ *= num_U_updates(level_params[j].integrator) * level_params[j].numsteps
            end

            trajectory / Nᵢ
        end

        numcv == 0 &&
            (@assert 0 ∉ forces "bias force cannot be in hmc level without bias")
        HMCLevel(
            integrator_from_str(lvl.integrator, rafriction, velocity, constraint),
            lvl.numsteps,
            Δτ,
            forces;
            numchildren,
            hmc_logging,
            logdir,
            instance,
            numcv,
            distributed=is_distributed(U),
        )
    end

    numsubsteps = 1
    sum_U_updates = 0
    switch = false
    for ilvl in length(levels):-1:1
        switch && continue
        lvl = levels[ilvl]
        int = level_params[ilvl].integrator
        if Val(0)in lvl.forces && ilvl == length(levels)
            switch = true
            numsubsteps = lvl.numsteps
        elseif Val(0) in lvl.forces
            switch = true
            numsubsteps = sum_U_updates * lvl.numsteps
        else
            sum_U_updates += lvl.numsteps * num_U_updates(int)
        end
    end

    substep_CVs = Vector{Float64}[]
    for _ in 1:numcv
        push!(substep_CVs, zeros(Int(numsubsteps)))
    end

    allforces = collect(Iterators.flatten([lvl.forces for lvl in levels]))
    fail = false

    if numcv > 0
        if Val(0) ∉ allforces
            @error("Bias force (i.e., force 0) not included in any level")
            fail = true
        end
    end

    if Val(1) ∉ allforces
        @error("Gauge force (i.e., force 1) not included in any level")
        fail = true
    end

    for ipf in 1:numfermions
        if Val(ipf + 1) ∉ allforces
            @error("Fermion force $(ipf) (i.e., force $(ipf+1)) not included in any level")
            fail = true
        end
    end

    for iforce in allforces
        if _unwrap_val(iforce) > numfermions+1
            @error("Force $(iforce) doesn't have a matching action. You probably don't have enough fermion actions.")
            fail = true
        end
    end

    fail && error("Forces missing")

    smearing_gauge = StoutSmearing(U; numlayers=numsmear_gauge, rho=rho_stout_gauge)
    smearing_fermion = if fermion_action == "quenched"
        NoSmearing()
    else
        StoutSmearing(U; numlayers=numsmear_fermion, rho=rho_stout_fermion)
    end

    has_smearing = smearing_gauge != NoSmearing() || smearing_fermion != NoSmearing()
    force2 = (!has_smearing && numcv == 0) ? nothing : Colorfield(U)

    if fermion_action == "staggered"
        ϕ = ntuple(_ -> Spinorfield(U; staggered=true, hw=1), numfermions)
    elseif fermion_action == "staggered_eo"
        ϕ = ntuple(_ -> even_odd(Spinorfield(U; staggered=true, hw=1)), numfermions)
    elseif fermion_action ∈ ["staggered_h1234", "staggered_1342"]
        ϕ = ntuple(_ -> Spinorfield(U; staggered=true, hw=2), numfermions)
    elseif fermion_action == "wilson"
        ϕ = ntuple(_ -> Spinorfield(U; hw=1), numfermions)
    elseif fermion_action == "wilson_eo"
        ϕ = ntuple(_ -> even_odd(Spinorfield(U; hw=2)), numfermions)
    elseif fermion_action ∈ ["quenched", "none"]
        ϕ = nothing
    else
        throw(AssertionError("Dynamical fermions \"$fermion_action\" not supported"))
    end

    fieldstrength = numcv > 0 ? Tensorfield(U) : nothing
    comm_instance = mpi_comm_instance()

    if hmc_logging && (logdir != "") && (!is_distributed(U) || mpi_amroot(comm_instance))
        # XXX: Probably want to swap this too in MPI PT-MetaD

        for ii in instance
            _logfile = joinpath(logdir, "hmc_acc_logs_$(lpad(ii, 3, "0")).txt")
            fp = fopen(_logfile, "w")
            printf(fp, StaticString("%-25s"), "ΔP2")
            printf(fp, StaticString("%-25s"), "ΔSg")
            printf(fp, StaticString("%-25s"), "ΔSf")
            printf(fp, StaticString("%-25s"), "ΔV")
            printf(fp, StaticString("%-25s"), "Work")
            printf(fp, StaticString("%-25s"), "ΔH")
            printf(fp, StaticString("%-25s"), "Total Action")
            printf(fp, StaticString("%-8s"), "Accepted")
            newline(fp)
            fclose(fp)
        end

        _logfile = joinpath(logdir, "hmc_acc_logs_$(lpad(instance[1], 3, "0")).txt")
        logfile = SStaticString(_logfile)
    else
        logfile = nothing
    end

    return HMC(
        levels,
        numlevels,
        friction,
        constraint,
        P,
        P_old,
        P0,
        U_old,
        U_high,
        U0,
        ϕ,
        staples,
        force,
        force2,
        fieldstrength,
        smearing_gauge,
        smearing_fermion,
        substep_CVs,
        logfile,
    )
end

include("hmc_integrators.jl")

function update!(
    hmc::HMC,
    U;
    fermion_action::TF=QuenchedFermionAction(),
    bias::TB=NoBias(),
    metro_test::Bool=true,
    therm::Val{THERM}=Val(false),
    instance::Int64=MPI_INSTANCE[],
    itrj=1,
) where {TF,TB,THERM}
    if TF !== QuenchedFermionAction
        @assert TF <: Tuple "fermion_action must be nothing or a tuple of fermion actions"
        @assert !isnothing(hmc.ϕ) "fermion_action passed but not activated in HMC"
    end

    hmc.substep_counter[] = 0

    set_ext!(hmc.logfile, instance)
    for lvl in hmc.levels
        set_ext!(lvl.forcefile, instance)
    end

    U_old = hmc.U_old
    # INFO: We always use double precision for the momenta and gauge action to improve reversibility
    U_high = isnothing(hmc.U_high) ? U : hmc.U_high 
    P_old = hmc.P_old
    P = hmc.P
    ϕ = hmc.ϕ
    smearing_gauge = hmc.smearing_gauge
    # Check if bias and fermion smearing have same parameters
    shared_smearing = (bias == NoBias()) ? false : (bias.smearing == hmc.smearing_fermion)
    smearing_fermion = shared_smearing ? bias.smearing : hmc.smearing_fermion
    friction = THERM ? 0.0 : hmc.friction
    numlevels = _unwrap_val(hmc.numlevels)

    if !isnothing(hmc.U_high)
        copy!(U_high, U)
        normalize!(U_high)
    end
    copy!(U_old, U)

    gaussian_TA!(P, friction)
    CV_old = calc_cv(U_high, bias)
    if is_constrained(hmc.levels[1]) && (!THERM || (!isnothing(hmc.constraint) && CV_old[1]!=3))
        therm = Val(false)
        enforce_hidden_constraint!(hmc, U, bias)
        CV_final = isnothing(hmc.constraint) ? CV_old[1] : hmc.constraint#get_sample(bias.sampler)
        hmc.levels[1].integrator.interval = (CV_final, CV_final)
        println(hmc.levels[1].integrator.interval)
    end
    !isnothing(P_old) && copy!(P_old, P)

    trP²_old = -calc_kinetic_energy(P)
    Sg_old = calc_gauge_action(U_high, smearing_gauge)
    V_old = bias(CV_old)
    sample_pseudofermions!(ϕ, fermion_action, U, smearing_fermion, shared_smearing)
    Sf_old = calc_fermion_action(fermion_action, U, ϕ, smearing_fermion, true) # INFO: fields are already smeared in sampling, so we dont have to here

    out = evolve!(U, hmc, fermion_action, bias, therm, numlevels)

    copy!(U_high, U)
    !isnothing(hmc.U_high) && normalize!(U_high)

    trP²_new = -calc_kinetic_energy(P)
    Sg_new = calc_gauge_action(U_high, smearing_gauge)
    CV_new = calc_cv(U_high, bias) # FIXME: this will error if there is not smearing in definition
    V_new = bias(CV_new)
    Sf_new = calc_fermion_action(fermion_action, U, ϕ, smearing_fermion, shared_smearing)
    work = if is_constrained(hmc.levels[1]) && !THERM
        0.0#sum(out[2])
    else
        0.0
    end

    ΔP² = trP²_new - trP²_old
    ΔSg = Sg_new - Sg_old
    ΔV = V_new - V_old
    ΔSf = Sf_new - Sf_old

    ΔH = ΔP² + ΔSg + ΔV + ΔSf + work
    S_new = Sg_new + V_new + Sf_new

    accept_root = metro_test ? rand() ≤ exp(-ΔH) : true

    accept = mpi_bcast_isbits(accept_root, mpi_comm_instance(); root=0)
    print_hmc_data(hmc.logfile, ΔP², ΔSg, ΔSf, ΔV, work, ΔH, S_new, accept)

    if accept
        set_cv!(bias, CV_new)
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

function updateU!(
    U::Gaugefield{B,T,M}, hmc, fac, fermion_action, bias, therm, level
) where {B,T,M}
    if level == 1
        ϵ = T(hmc.levels[level].Δτ * fac)
        P = hmc.P

        parallelfor(allindices(U, P), B, Val(M), (), (U,), (U, P)) do μsite, (U, P)
            # U[μsite] = cmatmul_oo(exp_iQ(-im * ϵ * P[μsite]), U[μsite])
            U[μsite] = proj_onto_SU3(cmatmul_oo(exp_iQ(-im * ϵ * P[μsite]), ComplexF64.(U[μsite])))
        end
        normalize!(U)
    else
        evolve!(U, hmc, fermion_action, bias, therm, level-1)
    end

    return nothing
end

function updateP!(U, hmc::HMC, fac, fermion_action, bias, level, recycle=false)
    lvl = hmc.levels[level]
    forces = lvl.forces
    ϵ = lvl.Δτ * fac
    P = hmc.P
    staples = hmc.staples
    force = hmc.force
    ϕ = hmc.ϕ
    temp_force = hmc.force2
    smearing_gauge = hmc.smearing_gauge
    if bias == NoBias() || isnothing(bias)
        shared_smearing = false
        smearing_fermion = hmc.smearing_fermion
    else
        shared_smearing = (bias.smearing == hmc.smearing_fermion)
        smearing_fermion = shared_smearing ? bias.smearing : hmc.smearing_fermion
    end

    fieldstrength = hmc.fieldstrength

    fp = !isnothing(lvl.forcefile) ? fopen(lvl.forcefile, "a") : nothing

    if Val(0) ∈ forces && !(lvl.integrator isa LeapfrogConstrained) && !(lvl.integrator isa OMF4Constrained)
        if bias isa Bias
            if recycle
                hmc.substep_counter[] += 1
            end
            substep_cv = hmc.substep_CVs_single
            for i in 1:length(bias)
                is_smeared = i > 1
                cv = calc_dVdU_bare!(
                    force, (fieldstrength, staples), U, temp_force, bias, i, is_smeared
                )
                substep_cv[i] = cv

                force_avg = norm(force, Val(2))
                force_sup = norm(force, Val(Inf))
                if !isnothing(fp)
                    # print(fp, cfmt("%+-25.15E", force_avg))
                    # print(fp, cfmt("%+-25.15E", force_sup))
                    printf(fp, StaticString("%+-25.15E"), force_avg)
                    printf(fp, StaticString("%+-25.15E"), force_sup)
                end

                add!(P, force, ϵ)
            end

            if recycle
                for i in eachindex(substep_cv)
                    hmc.substep_CVs[i][hmc.substep_counter[]] = substep_cv[i]
                end
            end
        else
            if !isnothing(fp)
                # print(fp, cfmt("%+-25.15E", 0.0))
                # print(fp, cfmt("%+-25.15E", 0.0))
                printf(fp, StaticString("%+-25.15E"), 0.0)
                printf(fp, StaticString("%+-25.15E"), 0.0)
            end
        end
    end

    if Val(1) ∈ forces
        calc_dSdU_bare!(force, staples, U, temp_force, smearing_gauge)

        force_avg = norm(force, Val(2))
        force_sup = norm(force, Val(Inf))
        if !isnothing(fp)
            # print(fp, cfmt("%+-25.15E", force_avg))
            # print(fp, cfmt("%+-25.15E", force_sup))
            printf(fp, StaticString("%+-25.15E"), force_avg)
            printf(fp, StaticString("%+-25.15E"), force_sup)
        end

        add!(P, force, ϵ)
    end

    if fermion_action !== QuenchedFermionAction()
        iforce = 0
        for i in forces
            (i == Val(0) || i == Val(1)) && continue # if bias or gauge force, go to next iteration
            iforce += 1

            is_smeared = (shared_smearing && Val(0) ∈ forces) || iforce > 1
            calc_dSfdU_bare!(
                force,
                fermion_action[_unwrap_val(i)-1],
                U,
                ϕ[_unwrap_val(i)-1],
                temp_force,
                smearing_fermion,
                is_smeared,
            )

            force_avg = norm(force, Val(2))
            force_sup = norm(force, Val(Inf))
            if !isnothing(fp)
                # print(fp, cfmt("%+-25.15E", force_avg))
                # print(fp, cfmt("%+-25.15E", force_sup))
                printf(fp, StaticString("%+-25.15E"), force_avg)
                printf(fp, StaticString("%+-25.15E"), force_sup)
            end

            add!(P, force, ϵ)
        end
    end

    if !isnothing(fp)
        newline(fp)
        fclose(fp)
        # print(fp, "\n")
        # close(fp)
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

sample_pseudofermions!(ϕ, ::QuenchedFermionAction, U, ::NoSmearing, ::Any) = nothing
sample_pseudofermions!(ϕ, ::QuenchedFermionAction, U, ::StoutSmearing, ::Any) = nothing

function sample_pseudofermions!(ϕ, fermion_action, U, ::NoSmearing, ::Any)
    for i in eachindex(fermion_action)
        sample_pseudofermions!(ϕ[i], fermion_action[i], U)
    end

    return nothing
end

# TODO: include numsmear as argument so we can lift the requirement that fermion smearing
# has to be exactly equal to cv smearing to save on that 
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

calc_fermion_action(::QuenchedFermionAction, U, ϕ, ::NoSmearing, ::Any) = 0.0
calc_fermion_action(::QuenchedFermionAction, U, ϕ, ::StoutSmearing, ::Any) = 0.0

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

@inline function print_hmc_data(::Nothing, ΔP², ΔSg, ΔSf, ΔV, work, ΔH, S, accept)
    @level2("delta_P²:\t$ΔP²")
    @level2("delta_Sg:\t$ΔSg")
    @level2("delta_Sf:\t$ΔSf")
    @level2("delta_V:\t$ΔV")
    @level2("W:\t$work")
    @level2("delta_H:\t$ΔH")
    @level2("new_S:\t$S")
    @level2("Accepted:\t$(Int64(accept))")
    return nothing
end

@inline function print_hmc_data(logfile, ΔP², ΔSg, ΔSf, ΔV, work, ΔH, S, accept)
    fp = fopen(logfile, "a")
    printf(fp, StaticString("%+-25.15E"), ΔP²)
    printf(fp, StaticString("%+-25.15E"), ΔSg)
    printf(fp, StaticString("%+-25.15E"), ΔSf)
    printf(fp, StaticString("%+-25.15E"), ΔV)
    printf(fp, StaticString("%+-25.15E"), work)
    printf(fp, StaticString("%+-25.15E"), ΔH)
    printf(fp, StaticString("%+-25.15E"), S)
    printf(fp, StaticString("%-i"), Int64(accept))
    newline(fp)
    fclose(fp)
    return nothing
end

# In order to write and load the bias easily with JLD2 for checkpointing, we need to define
# custom serialization, because saving and loading IOStreams doesn't work
using JLD2

struct HMCSerialization{TL,NL,TG,TT,TF,TSG,TSF,TPO,TF2,TFS,TLF}
    levels::TL
    numlevels::Val{NL}
    friction::Float64

    P::TT
    P_old::TPO # second momentum field for GHMC
    U_old::TG
    ϕ::TF
    staples::TT
    force::TT
    force2::TF2 # second force field for smearing
    fieldstrength::TFS # fieldstrength fields for Bias
    smearing_gauge::TSG
    smearing_fermion::TSF

    substep_CVs::Vector{Vector{Float64}}

    logfile::TLF
end

function JLD2.writeas(
    ::Type{<:HMC{TL,NL,TG,TT,TF,TSG,TSF,TPO,TF2,TFS,TFLS}}
) where {TL,NL,TG,TT,TF,TSG,TSF,TPO,TF2,TFS,TFLS}
    return HMCSerialization{TL,NL,TG,TT,TF,TSG,TSF,TPO,TF2,TFS,TFLS}
end

function Base.convert(::Type{<:HMCSerialization}, hmc::HMC)
    out = HMCSerialization(
        hmc.levels,
        hmc.numlevels,
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
        hmc.substep_CVs,
        hmc.logfile,
    )
    return out
end

function Base.convert(::Type{<:HMC}, hmc::HMCSerialization)
    out = HMC(
        hmc.levels,
        hmc.numlevels,
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
        hmc.substep_CVs,
        hmc.logfile,
    )
    return out
end
