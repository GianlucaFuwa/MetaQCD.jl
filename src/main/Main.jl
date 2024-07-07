module Main

using Dates
using DelimitedFiles
using InteractiveUtils
using MPI
using Random
using ..Output

import ..BiasModule: NoBias, calc_weights, recalc_CV!, update_bias!
import ..Fields: calc_gauge_action, normalize!
import ..Measurements: MeasurementMethods, calc_measurements, calc_measurements_flowed
import ..Parameters: construct_params_from_toml
import ..Smearing: GradientFlow
import ..Universe: Univ
import ..Updates: HMC, ParityUpdate, Updatemethod, update!, temper!

export build_bias, run_sim

const COMM = MPI.COMM_WORLD
const MYRANK = MPI.Comm_rank(COMM)
const COMM_SIZE = MPI.Comm_size(COMM)

const PACKAGE_VERSION = "1.0.0"

function print_acceptance_rates(numaccepts, itrj)
    MYRANK != 0 && return nothing

    for (i, value) in enumerate(numaccepts)
        @level1("|    Acceptance $i:\t$(100value / itrj) %")
    end

    return nothing
end

"""
Convenience-function to convert execution time from seconds to days
"""
function convert_seconds(sec)
    sec = round(Int, sec, RoundNearestTiesAway)
    x, seconds = divrem(sec, 60)
    y, minutes = divrem(x, 60)
    days, hours = divrem(y, 24)
    return "$(Day(days)), $(Hour(hours)), $(Minute(minutes)), $(Second(seconds))"
end

include("runbuild.jl")
include("runsim.jl")

end
