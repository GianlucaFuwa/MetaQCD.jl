module Updates

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using StaticArrays
using Polyester: @batch
using Printf
using Random: rand, default_rng
using StaticTools: StaticString
using Unicode
using ..MetaIO
using ..RHMCParameters
using ..Utils

import KernelAbstractions as KA
import ..BiasModule: Bias, NoBias, calc_cv, ∂V∂Q, recalc_cv!, set_cv!
import ..BiasModule: update_bias!, pack_buffer!, unpack_buffer!
import ..DiracOperators: AbstractDiracOperator, FermionAction, QuenchedFermionAction
import ..DiracOperators: calc_fermion_action, has_clover_term, sample_pseudofermions!
import ..Fields: AbstractGaugeAction, Gaugefield, Colorfield, identity_gauges!, get_global_dims
import ..Fields: WilsonGaugeAction, add!, calc_gauge_action, calc_kinetic_energy, update_halo!
import ..Fields: allindices, clear!, get_local_dims, normalize!, fieldstrength_eachsite!, float_type
import ..Fields: check_dims, even_odd, gaussian_TA!, mul!, staple, staple_eachsite!
import ..Fields: parallelfor, parallelfor_max, @latmap, @latsum, gauge_action
import ..Fields: AbstractField, Plaquette, Clover, Spinorfield, Tensorfield, is_distributed
import ..Forces: calc_dSdU_bare!, calc_dSfdU_bare!, calc_dVdU_bare!
import ..Parameters: ParameterSet
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing
import ..Smearing: calc_smearedU!, get_layer, stout_backprop!
import ..Universe: Univ

abstract type AbstractUpdate end

include("./heatbath.jl")
include("./hmc.jl")
include("./metropolis.jl")
include("./overrelaxation.jl")
include("./parity.jl")
include("./tempering.jl")
include("./instanton.jl")

function Updatemethod(parameters::ParameterSet, U; instance=mpi_myrank())
    updatemethod = Updatemethod(
        U,
        parameters.update_method,
        logdir=parameters.log_dir,
        fermion_action=parameters.fermion_action,
        num_fermions=length(parameters.fermions),
        num_cv=length(parameters.biases),
        metro_ϵ=parameters.metro_epsilon,
        metro_numhits=parameters.metro_numhits,
        metro_target_acc=parameters.metro_target_acc,
        hmc_levels=parameters.levels,
        hmc_trajectory=parameters.hmc_trajectory,
        hmc_friction=parameters.hmc_friction,
        hmc_rafriction=parameters.hmc_rafriction,
        hmc_numsmear_gauge=parameters.hmc_numsmear_gauge,
        hmc_numsmear_fermion=parameters.hmc_numsmear_fermion,
        hmc_rhostout_gauge=parameters.hmc_rhostout_gauge,
        hmc_rhostout_fermion=parameters.hmc_rhostout_fermion,
        hmc_logging=parameters.hmc_logging,
        hb_maxit=parameters.hb_maxit,
        numheatbath=parameters.numheatbath,
        or_algorithm=parameters.or_algorithm,
        numorelax=parameters.numorelax,
        instance=instance,
    )
    return updatemethod
end

function Updatemethod(
    U,
    update_method;
    logdir="",
    fermion_action="none",
    num_fermions=0,
    num_cv=0,
    metro_ϵ=0.1,
    metro_numhits=1,
    metro_target_acc=0.5,
    hmc_levels=DEFAULT_GAUGE_LEVEL,
    hmc_trajectory=1,
    hmc_friction=0,
    hmc_rafriction=0,
    hmc_numsmear_gauge=0,
    hmc_numsmear_fermion=0,
    hmc_rhostout_gauge=0,
    hmc_rhostout_fermion=0,
    hmc_logging=true,
    hb_maxit=1,
    numheatbath=1,
    or_algorithm="subgroups",
    numorelax=4,
    instance=mpi_myrank(),
)
    lower_case(str) = Unicode.normalize(str; casefold=true)
    if lower_case(update_method) == "hmc"
        updatemethod = HMC(
            U,
            hmc_levels,
            hmc_trajectory,
            hmc_friction,
            hmc_numsmear_gauge,
            hmc_numsmear_fermion,
            hmc_rhostout_gauge,
            hmc_rhostout_fermion;
            rafriction=hmc_rafriction,
            hmc_logging=hmc_logging,
            fermion_action=fermion_action,
            numfermions=num_fermions,
            numcv=num_cv,
            logdir=logdir,
            instance=instance,
        )
    elseif lower_case(update_method) == "metropolis"
        updatemethod = Metropolis(
            U, metro_ϵ, metro_numhits, metro_target_acc, or_algorithm, numorelax
        )
    elseif lower_case(update_method) == "heatbath"
        updatemethod = Heatbath(U, hb_maxit, numheatbath, or_algorithm, numorelax)
    else
        error("update method $(update_method) is not supported")
    end

    return updatemethod
end

const DEFAULT_GAUGE_LEVEL = [Dict(
    "integrator" => "Leapfrog",
    "forces" => [1],
    "numsteps" => 100,
)]

update!(::T, ::Any) where {T<:AbstractUpdate} = nothing
update!(::Nothing, ::Any) = nothing

Base.close(::T) where {T<:AbstractUpdate} = nothing

end
