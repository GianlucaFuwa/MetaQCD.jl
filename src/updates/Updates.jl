module Updates

using CUDA: i32
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
import ..Gaugefields: clear_U!, dims, normalize!, fieldstrength_eachsite!, float_type
import ..Gaugefields: gaussian_TA!, mul!, staple, staple_eachsite!, substitute_U!
import ..Gaugefields: @groupreduce, @latmap, @latsum, gauge_action
import ..Gaugefields: Abstractfield, Plaquette, Clover, Tensorfield
import ..BiasModule: Bias, calc_CV, ∂V∂Q, recalc_CV!
import ..BiasModule: kind_of_cv, update_bias!
import ..Parameters: ParameterSet
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing
import ..Smearing: calc_smearedU!, get_layer, stout_backprop!
import ..Universe: Univ

abstract type AbstractUpdate end

include("./derivatives.jl")
include("./heatbath.jl")
include("./hmc.jl")
include("./metropolis.jl")
include("./overrelaxation.jl")
include("./parity.jl")
include("./tempering.jl")

include("gpu_kernels/derivatives.jl")
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
    lower_case(str) = Unicode.normalize(str, casefold=true)
    if lower_case(update_method) == "hmc"
        updatemethod = HMC(U, hmc_integrator, hmc_trajectory, hmc_steps, hmc_friction,
                           hmc_numsmear, hmc_ρstout, verboselevel;
                           bias_enabled = kind_of_bias!="none", logdir=logdir)
    elseif lower_case(update_method) == "metropolis"
        updatemethod = Metropolis(U, eo, metro_ϵ, metro_numhits, metro_target_acc,
                                  or_algorithm, or_numOR)
    elseif lower_case(update_method) == "heatbath"
        updatemethod = Heatbath(U, eo, hb_MAXIT, hb_numHB, or_algorithm, or_numOR)
    else
        error("update method $(update_method) is not supported")
    end

    return updatemethod
end

update!(::T, U) where {T<:AbstractUpdate} = nothing
update!(::Nothing, U) = nothing

Base.close(::T) where {T<:AbstractUpdate} = nothing

end
