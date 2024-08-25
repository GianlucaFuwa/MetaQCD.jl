module Main

using Dates
using DelimitedFiles
using InteractiveUtils
using Random
using ..MetaIO
using ..Utils

import ..BiasModule: NoBias, calc_weights, recalc_CV!, update_bias!
import ..DiracOperators: fermaction_from_str
import ..Fields: calc_gauge_action, normalize!
import ..Measurements: MeasurementMethods, calc_measurements, calc_measurements_flowed
import ..Parameters: ParameterSet, construct_params_from_toml
import ..Smearing: GradientFlow
import ..Universe: Univ
import ..Updates: HMC, ParityUpdate, Updatemethod, update!, temper!, integrator_from_str

export build_bias, run_sim

const PACKAGE_VERSION = "1.0.0"

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
        "└\nTotal elapsed time:\t$days days $hours hours $minutes minutes $seconds seconds"
    )
end

include("runbuild.jl")
include("runsim.jl")

end
