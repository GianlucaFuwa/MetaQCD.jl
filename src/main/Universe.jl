module Universe

using Dates
using Unicode
using TOML: parsefile
using ..Utils
using ..Output

import ..DiracOperators: WilsonFermionAction
import ..DiracOperators: StaggeredFermionAction, StaggeredEOPreFermionAction
import ..Fields: Gaugefield, WilsonGaugeAction, IwasakiGaugeAction, DBW2GaugeAction
import ..Fields: SymanzikTreeGaugeAction
import ..BiasModule: Bias, NoBias
import ..Parameters: ParameterSet

const PACKAGE_VERSION = "1.0.0"

"""
    Univ(parameters::ParameterSet; use_mpi=false)

Create a Universe, containing the gauge configurations that are updated throughout a 
simulation and the bias potentials, if any. \\
`use_mpi` is soley an indicator that sets `numinstances` to 1 when using multiple walkers
even if stated otherwise in the parameters.
"""
struct Univ{TG,TF,TB}
    U::TG
    fermion_actions::TF
    bias::TB
    numinstances::Int64
end

function Univ(parameters::ParameterSet; use_mpi=false)
    @level1("[ Running MetaQCD.jl version $(PACKAGE_VERSION)\n")
    @level1("┌ Setting Universe...")
    @level1("|  MPI $(ifelse(use_mpi, "ENABLED", "DISABLED"))")
    @level1("|  NUM INSTANCES: $(parameters.numinstances)")
    NX, NY, NZ, NT = parameters.L
    β = parameters.beta
    backend = parameters.backend
    fp_prec = parameters.float_type
    gauge_action = parameters.gauge_action
    @level1("|  BACKEND: $(backend)")
    @level1("|  FP PREC: $(fp_prec)")
    @level1("|  L: $(NX)x$(NY)x$(NZ)x$(NT)")
    @level1("|  GAUGE ACTION: $(gauge_action)")
    @level1("|  BETA: $β")

    if parameters.kind_of_bias != "none"
        if parameters.tempering_enabled && use_mpi == false
            numinstances = parameters.numinstances
            U₁ = Gaugefield(parameters)
            fermion_actions₁ = init_fermion_actions(parameters, U₁)
            bias₁ = Bias(parameters, U₁; instance=0) # instance=0 -> dummy bias for non-MetaD stream

            U = Vector{typeof(U₁)}(undef, numinstances)
            fermion_actions = Vector{typeof(fermion_actions₁)}(undef, numinstances)
            bias = Vector{typeof(bias₁)}(undef, numinstances)

            U[1] = U₁
            fermion_actions[1] = fermion_actions₁
            bias[1] = bias₁
            for i in 2:numinstances
                U[i] = Gaugefield(parameters)
                fermion_actions[i] = deepcopy(fermion_actions₁)
                bias[i] = Bias(parameters, U[i]; instance=i - 1)
            end
        else
            numinstances = 1
            U = Gaugefield(parameters)
            fermion_actions = init_fermion_actions(parameters, U)
            bias = Bias(parameters, U; use_mpi=use_mpi)
        end
    else
        @assert parameters.tempering_enabled == false "tempering can only be enabled with bias"
        numinstances = 1
        U = Gaugefield(parameters)
        fermion_actions = init_fermion_actions(parameters, U)
        bias = NoBias()
    end

    @level1("└\n")
    return Univ{typeof(U),typeof(fermion_actions),typeof(bias)}(
        U, fermion_actions, bias, numinstances
    )
end

function init_fermion_actions(parameters::ParameterSet, U)
    fermion_action = parameters.fermion_action
    eo_precon = parameters.eo_precon
    Nf = parameters.Nf
    mass = parameters.mass
    @assert length(Nf) == length(mass) "Need same amount of masses as non-degenerate flavours"

    if fermion_action == "none"
        fermion_actions = nothing
    elseif fermion_action == "wilson"
        if eo_precon
            error("even-odd preconditioned wilson fermions not supported yet")
        else
            fermion_actions = ntuple(
                i -> WilsonFermionAction(
                    U,
                    mass[i];
                    Nf=Nf[i],
                    csw=parameters.wilson_csw,
                    r=parameters.wilson_r,
                    anti_periodic=parameters.anti_periodic,
                    cg_tol_action=parameters.cg_tol_action,
                    cg_tol_md=parameters.cg_tol_md,
                    cg_maxiters_action=parameters.cg_maxiters_action,
                    cg_maxiters_md=parameters.cg_maxiters_md,
                    rhmc_spectral_bound=parameters.rhmc_spectral_bound,
                    rhmc_order_action=parameters.rhmc_order_action,
                    rhmc_order_md=parameters.rhmc_order_md,
                    rhmc_prec_action=parameters.rhmc_prec_action,
                    rhmc_prec_md=parameters.rhmc_prec_md,
                ),
                length(Nf),
            )
        end
    elseif fermion_action == "staggered"
        if eo_precon
            fermion_actions = ntuple(
                i -> StaggeredEOPreFermionAction(
                    U,
                    mass[i];
                    Nf=Nf[i],
                    anti_periodic=parameters.anti_periodic,
                    cg_tol_action=parameters.cg_tol_action,
                    cg_tol_md=parameters.cg_tol_md,
                    cg_maxiters_action=parameters.cg_maxiters_action,
                    cg_maxiters_md=parameters.cg_maxiters_md,
                    rhmc_spectral_bound=parameters.rhmc_spectral_bound,
                    rhmc_order_action=parameters.rhmc_order_action,
                    rhmc_order_md=parameters.rhmc_order_md,
                    rhmc_prec_action=parameters.rhmc_prec_action,
                    rhmc_prec_md=parameters.rhmc_prec_md,
                ),
                length(Nf),
            )
        else
            fermion_actions = ntuple(
                i -> StaggeredFermionAction(
                    U,
                    mass[i];
                    Nf=Nf[i],
                    anti_periodic=parameters.anti_periodic,
                    cg_tol_action=parameters.cg_tol_action,
                    cg_tol_md=parameters.cg_tol_md,
                    cg_maxiters_action=parameters.cg_maxiters_action,
                    cg_maxiters_md=parameters.cg_maxiters_md,
                    rhmc_spectral_bound=parameters.rhmc_spectral_bound,
                    rhmc_order_action=parameters.rhmc_order_action,
                    rhmc_order_md=parameters.rhmc_order_md,
                    rhmc_prec_action=parameters.rhmc_prec_action,
                    rhmc_prec_md=parameters.rhmc_prec_md,
                ),
                length(Nf),
            )
        end
    else
        error("Fermion action \"$(fermion_action)\" not supported")
    end
    return fermion_actions
end

end
