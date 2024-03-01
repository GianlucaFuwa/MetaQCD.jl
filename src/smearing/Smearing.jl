"""
    AbstractSmearingModule

Module containing all smearing routines (stout for HMC/MetaD and gradflow for measurements)
Smearing types are subtypes of AbstractSmearing

    StoutSmearing -> struct holding temporary fields for stout smearing and stout force
                     recursion and the stout smearing parameters
                     is parametrized by TG, i.e. the type of Gaugefield and action

    GradientFlow -> struct holding temporary fields for flow and its parameters
                    is parametrized by the integrator, with supported integrators being
                    Euler, RK3 and RK3W7

"""
module Smearing

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester: @batch
using StaticArrays
using Unicode
using ..Output
using ..Utils

import ..Gaugefields: AbstractGaugeAction, CoeffField, Gaugefield, Temporaryfield
import ..Gaugefields: leftmul_dagg!, staple, staple_eachsite!, substitute_U!, @groupreduce
import ..Gaugefields: dims, float_type, i32, @latmap

abstract type AbstractSmearing end

struct NoSmearing <: AbstractSmearing end

include("./stout.jl")
include("./gradientflow.jl")

include("gpu_kernels/gradientflow.jl")
include("gpu_kernels/stout.jl")

function construct_smearing(U, smearingparameters, coefficient, numlayers)
    if smearingparameters == "nothing"
        smearing = NoSmearing()
    elseif smearingparameters == "stout"
        @assert coefficient !== nothing "Stout coefficient must be set"
        println("Stout smearing will be used")
        smearing = StoutSmearing(U, numlayers, coefficient)
    else
        error("Smearing = $smearing is not supported")
    end

    return smearing
end

calc_smearedU!(smearing::StoutSmearing, Uin) = apply_smearing!(smearing, Uin)
calc_smearedU!(smearing::GradientFlow, Uin) = flow!(smearing, Uin)

end
