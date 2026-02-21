module Universe

using Dates
using LinearAlgebra
using TOML: parsefile
using ..Logs
using ..Utils

import ..DiracOperators: FermionAction, QuenchedFermionAction, init_fermion_action
import ..Fields: Gaugefield, WilsonGaugeAction, IwasakiGaugeAction, DBW2GaugeAction
import ..Fields: SymanzikTreeGaugeAction, get_global_dims, num_floats
import ..BiasModule: Bias, NoBias
import ..Parameters: ParameterSet

"""
    Univ(parameters::ParameterSet; mpi_multi_sim=false)

Create a Universe, containing the gauge configurations that are updated throughout a 
simulation and the bias potentials, if any. \\
`mpi_multi_sim` is soley an indicator that sets `numinstances` to 1 when using
multiple walkers even if stated otherwise in the parameters.
"""
struct Univ{TG,TF,TB}
    U::TG
    fermion_action::TF
    bias::TB
    numinstances::Int64
    function Univ(
        U::Gaugefield{BACKEND,T,M,GA}, fermion_action::TF, bias::TB, numinstances
    ) where {BACKEND,T,M,GA,TF,TB}
        NX, NY, NZ, NT = size(U)
        @level1("- Constructing Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(string(BACKEND))")
        @level1("|  FP PREC: $(string(T))")
        @level1("|  L: $(NX)x$(NY)x$(NZ)x$(NT)")
        @level1("|  SU3 NUMFLOATS: $(num_floats(U))")
        @level1("|  GAUGE ACTION: $(string(GA))")
        @level1("|  BETA: $(Float64(U.β))")

        if TF === QuenchedFermionAction
            @level1("|  FERMION ACTION: Quenched\n-\n")
        else
            @level1("|  FERMION ACTION:\n$(string(fermion_action...))\n-\n")
        end

        TG = typeof(U)
        return new{TG,TF,TB}(U, fermion_action, bias, numinstances)
    end

    function Univ(
        U::Vector{TG}, fermion_action::TF, bias::Vector{TB}, numinstances
    ) where {B,T,M,GA,TG<:Gaugefield{B,T,M,GA},TF,TB}
        NX, NY, NZ, NT = size(U[1])
        @level1("- Constructing Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(string(B))")
        @level1("|  FP PREC: $(string(T))")
        @level1("|  L: $(NX)x$(NY)x$(NZ)x$(NT)")
        @level1("|  GAUGE ACTION: $(string(GA))")
        @level1("|  BETA: $(Float64(U[1].β))")

        if TF === QuenchedFermionAction
            @level1("|  FERMION ACTION:\n-\n")
        else
            @level1("|  FERMION ACTION:\n$(string(fermion_action...))-\n")
        end

        return new{Vector{TG},TF,Vector{TB}}(U, fermion_action, bias, numinstances)
    end
end

function Univ(parameters::ParameterSet; mpi_multi_sim=false, build=false)
    if length(parameters.biases) != 0
        if parameters.tempering_enabled && !mpi_multi_sim
            numinstances = parameters.numinstances
            U = [Gaugefield(parameters) for _ in 1:numinstances]
            dummy = parameters.meas_stream_bias ? false : (MPI_INSTANCE[]==0)
            bias = [Bias(parameters, U[i]; instance=i-1, dummy=i==1 ? dummy : false) for i in 1:numinstances]
            fermion_action = init_fermion_actions(parameters, U[1])
        elseif parameters.tempering_enabled && mpi_multi_sim
            numinstances = parameters.numinstances
            U = Gaugefield(parameters)
            fermion_action = init_fermion_actions(parameters, U)
            dummy = parameters.meas_stream_bias ? false : (MPI_INSTANCE[]==0)
            bias = Bias(parameters, U; mpi_multi_sim, dummy)
        else
            numinstances = 1
            U = Gaugefield(parameters)
            fermion_action = init_fermion_actions(parameters, U)
            bias = Bias(parameters, U; mpi_multi_sim, build)
        end
    else
        @assert parameters.tempering_enabled == false """
        tempering can only be enabled with bias
        """
        numinstances = 1
        U = Gaugefield(parameters)
        fermion_action = init_fermion_actions(parameters, U)
        bias = NoBias()
    end

    return Univ(U, fermion_action, bias, numinstances)
end

function init_fermion_actions(parameters::ParameterSet, U)
    fermions = parameters.fermions

    if length(fermions) == 0
        fermion_actions = QuenchedFermionAction()
    else
        fermion_actions = ntuple(length(fermions)) do i
            init_fermion_action(parameters, U, i)
        end
    end

    return fermion_actions
end

end
