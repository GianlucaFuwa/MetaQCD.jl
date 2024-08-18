module Smearing

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester: @batch
using StaticArrays
using Unicode
using ..Output
using ..Utils

import ..Fields: AbstractGaugeAction, Expfield, Colorfield, Gaugefield, WilsonGaugeAction
import ..Fields: check_dims, leftmul_dagg!, staple, staple_eachsite!, @groupreduce
import ..Fields: AbstractField, dims, float_type, @latmap

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
