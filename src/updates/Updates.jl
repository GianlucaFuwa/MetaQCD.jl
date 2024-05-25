module Updates

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using StaticArrays
using Polyester: @batch
using Printf
using Random: rand, default_rng
using Unicode
using ..Output
using ..Utils

import KernelAbstractions as KA
import ..Gaugefields: AbstractGaugeAction, Gaugefield, Temporaryfield
import ..Gaugefields: WilsonGaugeAction, add!, calc_gauge_action, calc_kinetic_energy
import ..Gaugefields: clear!, dims, normalize!, fieldstrength_eachsite!, float_type
import ..Gaugefields: check_dims, even_odd, gaussian_TA!, mul!, staple, staple_eachsite!
import ..Gaugefields: @groupreduce, @latmap, @latsum, gauge_action
import ..Gaugefields: Abstractfield, Plaquette, Clover, Fermionfield, Tensorfield
import ..DiracOperators: AbstractDiracOperator, calc_fermion_action, fermaction_from_str
import ..DiracOperators: sample_pseudofermions!
import ..BiasModule: Bias, calc_CV, ∂V∂Q, recalc_CV!
import ..BiasModule: kind_of_cv, update_bias!
import ..Parameters: ParameterSet
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing
import ..Smearing: calc_smearedU!, get_layer, stout_backprop!
import ..Universe: Univ

abstract type AbstractUpdate end

include("../forces/forces.jl")
include("./heatbath.jl")
include("./hmc.jl")
include("./metropolis.jl")
include("./overrelaxation.jl")
include("./parity.jl")
include("./tempering.jl")

include("gpu_kernels/heatbath.jl")
include("gpu_kernels/hmc.jl")
include("gpu_kernels/metropolis.jl")
include("gpu_kernels/overrelaxation.jl")
include("gpu_kernels/parity.jl")
include("gpu_kernels/tempering.jl")

function Updatemethod(parameters::ParameterSet, U)
    updatemethod = Updatemethod(
        U,
        parameters.update_method,
        parameters.logdir,
        parameters.fermion_action,
        parameters.eo_precon,
        parameters.Nf,
        parameters.kind_of_bias,
        parameters.metro_epsilon,
        parameters.metro_numhits,
        parameters.metro_target_acc,
        parameters.hmc_integrator,
        parameters.hmc_steps,
        parameters.hmc_trajectory,
        parameters.hmc_friction,
        parameters.hmc_numsmear_gauge,
        parameters.hmc_numsmear_fermion,
        parameters.hmc_rhostout_gauge,
        parameters.hmc_rhostout_fermion,
        parameters.hmc_logging,
        parameters.hb_maxit,
        parameters.numheatbath,
        parameters.eo,
        parameters.or_algorithm,
        parameters.numorelax,
    )
    return updatemethod
end

function Updatemethod(
    U,
    update_method,
    logdir="",
    fermion_action="none",
    eo_precon=false,
    Nf=0,
    kind_of_bias="none",
    metro_ϵ=0.1,
    metro_numhits=1,
    metro_target_acc=0.5,
    hmc_integrator="leapfrog",
    hmc_steps=10,
    hmc_trajectory=1,
    hmc_friction=π / 2,
    hmc_numsmear_gauge=0,
    hmc_numsmear_fermion=0,
    hmc_rhostout_gauge=0,
    hmc_rhostout_fermion=0,
    hmc_logging=true,
    hb_MAXIT=1,
    hb_numHB=1,
    eo=true,
    or_algorithm="subgroups",
    or_numOR=4,
)
    lower_case(str) = Unicode.normalize(str; casefold=true)
    if lower_case(update_method) == "hmc"
        updatemethod = HMC(
            U,
            integrator_from_str(hmc_integrator),
            hmc_trajectory,
            hmc_steps,
            hmc_friction,
            hmc_numsmear_gauge,
            hmc_numsmear_fermion,
            hmc_rhostout_gauge,
            hmc_rhostout_fermion;
            hmc_logging=hmc_logging,
            fermion_action=fermaction_from_str(lower_case(fermion_action), eo_precon),
            heavy_flavours=length(Nf) - 1,
            bias_enabled=kind_of_bias != "none",
            logdir=logdir,
        )
    elseif lower_case(update_method) == "metropolis"
        updatemethod = Metropolis(
            U, eo, metro_ϵ, metro_numhits, metro_target_acc, or_algorithm, or_numOR
        )
    elseif lower_case(update_method) == "heatbath"
        updatemethod = Heatbath(U, eo, hb_MAXIT, hb_numHB, or_algorithm, or_numOR)
    else
        error("update method $(update_method) is not supported")
    end

    return updatemethod
end

update!(::T, ::Any) where {T<:AbstractUpdate} = nothing
update!(::Nothing, ::Any) = nothing

Base.close(::T) where {T<:AbstractUpdate} = nothing

end
