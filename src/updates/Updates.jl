module Updates

using Base.Threads: @threads, nthreads, threadid
using LinearAlgebra
using StaticArrays
using Polyester
using Printf
using Random: rand, default_rng
using ..Output
using ..Utils

import ..Gaugefields: AbstractGaugeAction, Gaugefield, Liefield, Temporaryfield
import ..Gaugefields: WilsonGaugeAction, add!, calc_gauge_action, calc_kinetic_energy,
    clear_U!, normalize!, fieldstrength_eachsite!, gaussian_momenta!, mul!, staple,
    staple_eachsite!, substitute_U!
import ..Gaugefields: Plaquette, Clover
import ..BiasModule: Bias, calc_CV, ∂V∂Q, recalc_CV!
import ..BiasModule: kind_of_cv, update_bias!
import ..Parameters: ParameterSet
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing
import ..Smearing: calc_smearedU!, get_layer, stout_backprop!
import ..Universe: Univ

abstract type AbstractUpdate end

include("./hmc.jl")
include("./metropolis.jl")
include("./heatbath.jl")
include("./overrelaxation.jl")
include("./parity.jl")
include("./tempering.jl")

function Updatemethod(parameters::ParameterSet, U, verbose)
    updatemethod = Updatemethod(
        U,
        parameters.update_method,
        verbose,
        parameters.verboselevel,
        parameters.logdir,
        parameters.kind_of_bias,
        parameters.metro_epsilon,
        parameters.metro_numhits,
        parameters.metro_target_acc,
        parameters.hmc_integrator,
        parameters.hmc_steps,
        parameters.hmc_trajectory,
        parameters.hmc_friction,
        parameters.hmc_numsmear,
        parameters.hmc_rhostout,
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
    verbose,
    verboselevel = 1,
    logdir = "",
    kind_of_bias = "none",
    metro_ϵ = 0.1,
    metro_numhits = 1,
    metro_target_acc = 0.5,
    hmc_integrator = "Leapfrog",
    hmc_steps = 10,
    hmc_trajectory = 1,
    hmc_friction = π/2,
    hmc_numsmear = 0,
    hmc_ρstout = 0,
    hb_MAXIT = 1,
    hb_numHB = 1,
    eo = true,
    or_algorithm = "subgroups",
    or_numOR = 4,
)
    if update_method == "hmc"
        updatemethod = HMC(
            U, hmc_integrator, hmc_steps, hmc_trajectory;
            verbose = verbose,
            friction = hmc_friction,
            numsmear = hmc_numsmear,
            ρ_stout = hmc_ρstout,
            bias_enabled = kind_of_bias!="none",
            verboselevel = verboselevel,
            logdir = logdir,
        )
    elseif update_method == "metropolis"
        updatemethod = Metropolis(
            U, eo, metro_ϵ, metro_numhits, metro_target_acc, or_algorithm, or_numOR;
            verbose = verbose,
        )
    elseif update_method == "heatbath"
        updatemethod = Heatbath(
            U, eo, hb_MAXIT, hb_numHB, or_algorithm, or_numOR;
            verbose = verbose,
        )
    else
        error("update method $(update_method) is not supported")
    end

    return updatemethod
end

update!(::T, U) where {T<:AbstractUpdate} = nothing
update!(::Nothing, U) = nothing

end
