module Universe

using Dates
using LinearAlgebra
using Unicode
using TOML: parsefile
using ..MetaIO
using ..Utils

import ..DiracOperators: FermionAction, QuenchedFermionAction, init_fermion_action
import ..Fields: Gaugefield, WilsonGaugeAction, IwasakiGaugeAction, DBW2GaugeAction
import ..Fields: SymanzikTreeGaugeAction
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
        U::Gaugefield{BACKEND,T,M,A,GA}, fermion_action::TF, bias::TB, numinstances
    ) where {BACKEND,T,M,A,GA,TF,TB}
        @level1("- Constructing Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(string(BACKEND))")
        @level1("|  FP PREC: $(string(T))")
        @level1("|  L: $(U.NX)x$(U.NY)x$(U.NZ)x$(U.NT)")
        @level1("|  GAUGE ACTION: $(string(GA))")
        @level1("|  BETA: $(U.β)")

        if TF === QuenchedFermionAction
            @level1("|  FERMION ACTION: Quenched\n-\n")
        else
            @level1("|  FERMION ACTION: $(string(fermion_action...))\n-\n")
        end

        TG = typeof(U)
        return new{TG,TF,TB}(U, fermion_action, bias, numinstances)
    end

    function Univ(
        U::Vector{TG}, fermion_action::TF, bias::Vector{TB}, numinstances
    ) where {B,T,M,A,GA,TG<:Gaugefield{B,T,M,A,GA},TF,TB}
        @level1("- Constructing Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(string(B))")
        @level1("|  FP PREC: $(string(T))")
        @level1("|  L: $(U[1].NX)x$(U[1].NY)x$(U[1].NZ)x$(U[1].NT)")
        @level1("|  GAUGE ACTION: $(string(GA))")
        @level1("|  BETA: $(U[1].β)")

        if TF === QuenchedFermionAction
            @level1("|  FERMION ACTION:\n-\n")
        else
            @level1("|  FERMION ACTION: $(string(fermion_action...))-\n")
        end

        return new{Vector{TG},TF,Vector{TB}}(U, fermion_action, bias, numinstances)
    end
end

function Univ(parameters::ParameterSet; mpi_multi_sim=false, build=false)
    if length(parameters.biases) != 0
        if parameters.tempering_enabled && !mpi_multi_sim
            numinstances = parameters.numinstances
            U₁ = Gaugefield(parameters)
            fermion_action = init_fermion_actions(parameters, U₁)
            bias₁ = Bias(parameters, U₁; dummy=true) # dummy bias for non-MetaD stream

            U = Vector{typeof(U₁)}(undef, numinstances)
            bias = Vector{Bias}(undef, numinstances) # XXX: Type unstable
            U[1] = U₁
            bias[1] = bias₁

            for i in 2:numinstances
                U[i] = Gaugefield(parameters)
                bias[i] = Bias(parameters, U[i]; instance=i-1)
            end
        elseif parameters.tempering_enabled && mpi_multi_sim
            numinstances = 1
            U = Gaugefield(parameters)
            fermion_action = init_fermion_actions(parameters, U)
            bias = Bias(parameters, U; mpi_multi_sim=mpi_multi_sim, dummy=MPI_INSTANCE[]==0)
        else
            numinstances = 1
            U = Gaugefield(parameters)
            fermion_action = init_fermion_actions(parameters, U)
            bias = Bias(parameters, U; mpi_multi_sim=mpi_multi_sim, build=build)
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
