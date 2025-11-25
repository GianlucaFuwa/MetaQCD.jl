"""
    MeasurementModule

Module containing all measurement methods and their parameters. In particular, all
observables get a struct (subtype of `AbstractMeasurement`) with its parameters, file name
and so on. We then define a `measure` function for each observable which calculates the
observable and prints it to file and/or console.
"""
module Measurements

using LinearAlgebra
using Polyester
using ..Logs
using ..Utils

import ..DiracOperators: Daggered, DdaggerD, StaggeredDiracOperator, WilsonDiracOperator
import ..DiracOperators: StaggeredEOPreDiracOperator, even_odd, solve_dirac!
import ..DiracOperators: ArnoldiWorkspaceMeta, get_eigenvalues, num_dirac
import ..DiracOperators: FermionAction, calc_fermion_action, sample_pseudofermions!
import ..Fields: WilsonGaugeAction, SymanzikTreeGaugeAction, SymanzikTadGaugeAction
import ..Fields: IwasakiGaugeAction, DBW2GaugeAction, AbstractFieldstrength, Colorfield
import ..Fields: Gaugefield, Spinorfield, Tensorfield, calc_gauge_action, check_dims 
import ..Fields: Plaquette, Clover, Improved, fieldstrength_eachsite!, gauge_action, staple
import ..Fields: clover_1x1, clover_2x1, clover_1x2, float_type, plaquette, wilsonloop
import ..Fields: parallelfor, parallelfor_sum, CPU, ones!, set_source!
import ..Fields: clear!, distributed_reduce, is_distributed, plaquette_trace_sum
import ..Fields: update_halo!, get_local_dims, get_global_dims, get_global_volume
import ..Smearing: AbstractSmearing, Cooling, GradientFlow, NoSmearing, StoutSmearing
import ..Smearing: calc_smearedU!, flow!

abstract type AbstractMeasurement end

const MYEXT_str = "_$(lpad(mpi_myrank(), 3, "0")).txt"
const MYEXT = StaticString(MYEXT_str)

@inline function digit_to_char(x::Integer)
    @assert 10 > x ≥ 0
    return Char('0' + x)
end

const ITRJ_STR_FMT = StaticString("%-11s")
const IFLOW_STR_FMT = StaticString("%-7s")
const TFLOW_STR_FMT = StaticString("%-9s")
const METHOD_STR_FMT = StaticString("%-25s")
const ITRJ_FMT = StaticString("%-11i")
const IFLOW_FMT = StaticString("%-7i")
const TFLOW_FMT = StaticString("%-9.5f")
const METHOD_FMT = StaticString("%-25.15E")

function measure(::M, args...; kwargs...) where {M<:AbstractMeasurement}
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
include("measure_logdet.jl")
# include("measure_eigenvalues.jl")

end
