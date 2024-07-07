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
using MPI
using Polyester
using Printf
using Unicode
using ..Output
using ..Utils

import KernelAbstractions as KA
import ..DiracOperators: Daggered, DdaggerD, StaggeredDiracOperator, WilsonDiracOperator
import ..DiracOperators: StaggeredEOPreDiracOperator, even_odd, solve_dirac!
import ..DiracOperators: ArnoldiWorkspaceMeta, get_eigenvalues
import ..Fields: WilsonGaugeAction, SymanzikTreeGaugeAction, SymanzikTadGaugeAction
import ..Fields: IwasakiGaugeAction, DBW2GaugeAction
import ..Fields: Gaugefield, Fermionfield, calc_gauge_action, clover_rect, clear!
import ..Fields: clover_square, dims, float_type, plaquette, wilsonloop, set_source!
import ..Fields: @groupreduce, @latsum, Plaquette, Clover, Improved, CPU, ones!
import ..Fields: check_dims, plaquette_trace_sum, wilsonloop
import ..Smearing: StoutSmearing, calc_smearedU!, flow!

abstract type AbstractMeasurement end

const MYRANK = MPI.Comm_rank(MPI.COMM_WORLD)

function Base.close(m::AbstractMeasurement)
    if hasfield(typeof(m), :fp)
        typeof(m.fp) == IOStream && close(m.fp)
    end
    return nothing
end

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
    return error("measurement with a type $M is not supported")
end

@inline get_myinstance(m::M) where {M<:AbstractMeasurement} = m.myinstance[]

@inline function set_myinstance!(m::M, new_instance) where {M<:AbstractMeasurement}
    m.myinstance[] = new_instance
end

include("measure_gauge_action.jl")
include("measure_plaquette.jl")
include("measure_polyakov.jl")
include("measure_wilson_loop.jl")
include("measure_topological_charge.jl")
include("measure_energy_density.jl")
include("measure_pion_correlator.jl")
include("measure_eigenvalues.jl")

include("gpu_kernels/energydensity.jl")
include("gpu_kernels/polyakov.jl")
include("gpu_kernels/topcharge.jl")

end
