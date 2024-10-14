"""
    MeasurementModule

Module containing all measurement methods and their parameters. In particular, all
observables get a struct (subtype of `AbstractMeasurement`) with its parameters, file name
and so on. We then define a `measure` function for each observable which calculates the
observable and prints it to file and/or console.
"""
module Measurements

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester
using Printf: @sprintf
using StaticTools: StaticString
using Unicode
using ..MetaIO
using ..Utils

import KernelAbstractions as KA
import ..DiracOperators: Daggered, DdaggerD, StaggeredDiracOperator, WilsonDiracOperator
import ..DiracOperators: StaggeredEOPreDiracOperator, even_odd, solve_dirac!
import ..DiracOperators: ArnoldiWorkspaceMeta, get_eigenvalues, num_dirac
import ..Fields: WilsonGaugeAction, SymanzikTreeGaugeAction, SymanzikTadGaugeAction
import ..Fields: IwasakiGaugeAction, DBW2GaugeAction, AbstractFieldstrength
import ..Fields: Gaugefield, Spinorfield, calc_gauge_action, check_dims, clover_rect, clear! 
import ..Fields: clover_square, global_dims, local_dims, float_type, plaquette, wilsonloop
import ..Fields: @groupreduce, @latsum, Plaquette, Clover, Improved, CPU, ones!, set_source!
import ..Fields: distributed_reduce, is_distributed, plaquette_trace_sum, wilsonloop
import ..Smearing: StoutSmearing, calc_smearedU!, flow!

abstract type AbstractMeasurement end

const MYEXT_str = "_$(lpad("$(mpi_myrank())", 4, "0")).txt"
const MYEXT = StaticString(MYEXT_str)

@inline set_ext!(filename::String, ::Integer) = filename

@inline function set_ext!(filename::StaticString, myinstance::Integer)
    filename[end-5] = digit_to_char(myinstance)
    return filename
end

@inline function digit_to_char(x::Integer)
    @assert x ≥ 0
    return Char('0' + x)
end

function measure(::M, args...) where {M<:AbstractMeasurement}
    return error("Measurement of type $M is not supported")
end

include("./measurement_parameters.jl")
include("./measurement_methods.jl")

include("measure_gauge_action.jl")
include("measure_plaquette.jl")
include("measure_polyakov.jl")
include("measure_wilson_loop.jl")
include("measure_topological_charge.jl")
include("measure_energy_density.jl")
include("measure_pion_correlator.jl")
# include("measure_eigenvalues.jl")

include("gpu_kernels/energydensity.jl")
include("gpu_kernels/polyakov.jl")
include("gpu_kernels/topcharge.jl")

end
