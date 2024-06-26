"""
    MeasurementModule

Module containing all measurement methods and their parameters. Also handles I/O

    MeasurementParameters -> Holds the parameters for each measurement supported, like
                             the interval of measurement and their methods (for TC and GA)

    MeasurementMethods -> Holds all measurements specified in parameter file and is accessed
                          in mainrun with 'calc_measurements' and 'calc_measurements_flowed'
"""
module Measurements

using CUDA: i32
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using MPI
using Polyester
using Printf
using Unicode
using ..Output
using ..Utils

import KernelAbstractions as KA
import ..Gaugefields: Gaugefield, calc_gauge_action, clover_rect, clover_square, dims,
    float_type, plaquette, wilsonloop, substitute_U!, @groupreduce, @latsum
import ..Gaugefields: Plaquette, Clover, Improved, CPU
import ..Smearing: StoutSmearing, calc_smearedU!, flow!

abstract type AbstractMeasurement end

Base.close(m::AbstractMeasurement) = typeof(m.fp)==IOStream ? close(m.fp) : nothing

include("./measurement_parameters.jl")
include("./measurement_methods.jl")

struct MeasurementOutput{T}
    value::T
    outputstring::String

    MeasurementOutput(value, str) = new{typeof(value)}(value, str)
end

get_value(m::MeasurementOutput) = m.value
get_string(m::MeasurementOutput) = m.outputstring

function measure(::M, args...) where {M<:AbstractMeasurement}
    error("measurement with a type $M is not supported")
end

include("measure_gauge_action.jl")
include("measure_plaquette.jl")
include("measure_polyakov.jl")
include("measure_wilson_loop.jl")
include("measure_topological_charge.jl")
include("measure_energy_density.jl")

include("gpu_kernels/energydensity.jl")
include("gpu_kernels/polyakov.jl")
include("gpu_kernels/topcharge.jl")

end
