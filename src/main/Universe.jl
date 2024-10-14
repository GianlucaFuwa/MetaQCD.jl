module Universe

using Dates
using Unicode
using TOML: parsefile
using ..MetaIO
using ..Utils

import ..DiracOperators: WilsonFermionAction, WilsonEOPreFermionAction
import ..DiracOperators: StaggeredFermionAction, StaggeredEOPreFermionAction
import ..DiracOperators: init_fermion_action, QuenchedFermionAction
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
    myinstance::Base.RefValue{Int64}
    numinstances::Int64
    function Univ(
        U::Gaugefield{BACKEND,T,A,GA}, fermion_action::TF, bias::TB, numinstances
    ) where {BACKEND,T,A,GA,TF,TB}
        @level1("┌ Setting Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(string(BACKEND))")
        @level1("|  FP PREC: $(string(T))")
        @level1("|  L: $(U.NX)x$(U.NY)x$(U.NZ)x$(U.NT)")
        @level1("|  GAUGE ACTION: $(string(GA))")
        @level1("|  BETA: $(U.β)")

        if TF === QuenchedFermionAction
            @level1("|  FERMION ACTION: Quenched\n└\n")
        else
            @level1("|  FERMION ACTION: $(string(fermion_action...))\n└\n")
        end

        TG = typeof(U)
        myinstance = Base.RefValue{Int64}(mpi_myrank())
        return new{TG,TF,TB}(U, fermion_action, bias, myinstance, numinstances)
    end

    function Univ(
        U::Vector{Gaugefield{BACKEND,T,A,GA}}, fermion_action::TF, bias::TB, numinstances
    ) where {BACKEND,T,A,GA,TF,TB<:Vector{Bias}}
        @level1("┌ Setting Universe...")
        @level1("|  NUM INSTANCES: $(numinstances)")
        @level1("|  BACKEND: $(BACKEND)")
        @level1("|  FP PREC: $(T)")
        @level1("|  L: $(U[1].NX)x$(U[1].NY)x$(U[1].NZ)x$(U[1].NT)")
        @level1("|  GAUGE ACTION: $(GA)")
        @level1("|  BETA: $(U[1].β)")

        if TF === QuenchedFermionAction
            @level1("|  FERMION ACTION:\n└\n")
        else
            @level1("|  FERMION ACTION: $(fermion_action...)└\n")
        end

        TG = typeof(U)
        myinstance = Base.RefValue{Int64}(mpi_myrank())
        return new{TG,TF,TB}(U, fermion_action, bias, myinstance, numinstances)
    end
end

function Univ(parameters::ParameterSet; mpi_multi_sim=false)
    if parameters.kind_of_bias != "none"
        if parameters.tempering_enabled && !mpi_multi_sim
            numinstances = parameters.numinstances
            U₁ = Gaugefield(parameters)
            fermion_action = init_fermion_actions(parameters, U₁)
            bias₁ = Bias(parameters, U₁; instance=0) # instance=0 -> dummy bias for non-MetaD stream

            U = Vector{typeof(U₁)}(undef, numinstances)
            bias = Vector{typeof(bias₁)}(undef, numinstances)
            U[1] = U₁
            bias[1] = bias₁

            for i in 2:numinstances
                U[i] = Gaugefield(parameters)
                bias[i] = Bias(parameters, U[i]; instance=i - 1)
            end
        else
            numinstances = 1
            U = Gaugefield(parameters)
            fermion_action = init_fermion_actions(parameters, U)
            bias = Bias(parameters, U; mpi_multi_sim=mpi_multi_sim)
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
    fermion_action = parameters.fermion_action
    Nf = parameters.Nf
    mass = parameters.mass
    @assert length(Nf) == length(mass) "Need same amount of masses as unique flavors"

    if fermion_action ∈ ("none", "quenched")
        fermion_actions = QuenchedFermionAction()
    else
        fermion_actions = ntuple(
            i -> init_fermion_action(parameters, mass[i], Nf[i], U), length(Nf),
        )
    end

    return fermion_actions
end

end
