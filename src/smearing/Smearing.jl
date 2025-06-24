module Smearing

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester: @batch
using StaticArrays
using Unicode
using ..MetaIO
using ..Utils

import ..Fields: AbstractGaugeAction, Expfield, Colorfield, Gaugefield, WilsonGaugeAction
import ..Fields: check_dims, leftmul_dagg!, staple, staple_eachsite!, update_halo!, size
import ..Fields: AbstractField, get_local_dims, float_type, gauge_action, @groupreduce, @latmap

abstract type AbstractSmearing end

struct NoSmearing <: AbstractSmearing end

include("./stout.jl")
include("./gradientflow.jl")
include("./cooling.jl")
include("gpu_kernels/gradientflow.jl")
include("gpu_kernels/stout.jl")
include("gpu_kernels/cooling.jl")

function construct_flow(U, parameters)
    flow_integrator = lower_case.(parameters.flow_integrator)

    # can measure using multiple integrators in one simulation
    smearing = ntuple(length(flow_integrator)) do i
        if flow_integrator[i] == "none"
            NoSmearing()
        elseif flow_integrator[i] == "cooling"
            Cooling(
                U;
                numflow=parameters.flow_num,
                measure_every=parameters.flow_measure_every,
            )
        elseif flow_integrator[i] ∈ ("euler", "rk2", "rk3", "rk3w7")
            GradientFlow(
                U;
                integrator=parameters.flow_integrator[i],
                numflow=parameters.flow_num,
                steps=parameters.flow_steps,
                tf=parameters.flow_tf,
                measure_every=parameters.flow_measure_every,
            )
        else
            error("Flow integrator \"$(parameters.flow_integrator)\" is not supported")
        end
    end

    return smearing
end

calc_smearedU!(::NoSmearing, ::Any) = nothing
calc_smearedU!(smearing::StoutSmearing, Uin) = apply_smearing!(smearing, Uin)
calc_smearedU!(smearing::GradientFlow, Uin) = flow!(smearing, Uin)
calc_smearedU!(smearing::Cooling, Uin) = flow!(smearing, Uin)

end
