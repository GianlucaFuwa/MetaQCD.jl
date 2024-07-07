module Universe

using Dates
using MPI
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

const COMM = MPI.COMM_WORLD
const MYRANK = MPI.Comm_rank(COMM)
const COMM_SIZE = MPI.Comm_size(COMM)

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
    myinstance::Base.RefValue{Int64}
    numinstances::Int64
    function Univ(
        U::Gaugefield{BACKEND,T,A,GA}, fermion_actions::TF, bias::TB, numinstances
    ) where {BACKEND,T,A,GA,TF<:Tuple,TB}
        @level1("┌ Setting Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(BACKEND)")
        @level1("|  FP PREC: $(T)")
        @level1("|  L: $(U.NX)x$(U.NY)x$(U.NZ)x$(U.NT)")
        @level1("|  GAUGE ACTION: $(GA)")
        @level1("|  BETA: $(U.β)")
        if TF === Tuple{}
            @level1("|  FERMION ACTIONS:\n└\n")
        else
            @level1("|  FERMION ACTIONS: $(fermion_actions...)└\n")
        end
        TG = typeof(U)
        return new{TG,TF,TB}(U, fermion_actions, bias, Base.RefValue{MYRANK}, numinstances)
    end

    function Univ(
        U::Vector{Gaugefield{BACKEND,T,A,GA}}, fermion_actions::TF, bias::TB, numinstances
    ) where {BACKEND,T,A,GA,TF<:Tuple,TB}
        @level1("┌ Setting Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(BACKEND)")
        @level1("|  FP PREC: $(T)")
        @level1("|  L: $(U[1].NX)x$(U[1].NY)x$(U[1].NZ)x$(U[1].NT)")
        @level1("|  GAUGE ACTION: $(GA)")
        @level1("|  BETA: $(U[1].β)")
        if TF === Tuple{}
            @level1("|  FERMION ACTIONS:\n└\n")
        else
            @level1("|  FERMION ACTIONS: $(fermion_actions...)└\n")
        end
        TG = typeof(U)
        return new{TG,TF,TB}(U, fermion_actions, bias, Base.RefValue{MYRANK}, numinstances)
    end
end

function Univ(parameters::ParameterSet; use_mpi=false)
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

    return Univ(U, fermion_actions, bias, Base.RefValue{MYRANK}, numinstances)
end

function init_fermion_actions(parameters::ParameterSet, U)
    fermion_action = parameters.fermion_action
    eo_precon = parameters.eo_precon
    Nf = parameters.Nf
    mass = parameters.mass
    @assert length(Nf) == length(mass) "Need same amount of masses as non-degenerate flavours"

    if fermion_action == "none"
        fermion_actions = Tuple{}
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
