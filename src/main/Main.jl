module Main

using Dates
using DelimitedFiles
using InteractiveUtils
using LinearAlgebra
using Random
using Statistics
using StaticTools: StaticString
using ..MetaIO
using ..Utils
using ..Fields: BACKENDS

import ..BiasModule: NoBias, calc_weights, is_adaptive, recalc_cv!, update_bias!, set_sigma0!
import ..DiracOperators: QuenchedFermionAction
import ..Fields: calc_gauge_action, is_distributed, normalize!
import ..Measurements: MeasurementMethods, calc_measurements, calc_measurements_flowed
import ..Parameters: ParameterSet, construct_params_from_toml
import ..Smearing: construct_flow
import ..Universe: Univ
import ..Updates: HMC, ParityUpdate, Updatemethod, update!, temper!, integrator_from_str
import ..Updates: set_instanton!

export build_bias, run_sim

const PACKAGE_VERSION = "2.0.0"

function print_acceptance_rates(numaccepts, itrj)
    for (i, value) in enumerate(numaccepts)
        @level1("|    Acceptance $i:\t$(100value / itrj) %")
    end

    return nothing
end

function print_total_time(sec::AbstractFloat)
    sec = round(Int64, sec, RoundNearestTiesAway)
    x, seconds = divrem(sec, 60)
    y, minutes = divrem(x, 60)
    days, hours = divrem(y, 24)
    @level1(
        ">> Total elapsed time:\t$days days $hours hours $minutes minutes $seconds seconds"
    )
end

include("runbuild.jl")
include("runsim.jl")

end
