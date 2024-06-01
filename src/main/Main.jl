module Main

using Dates
using DelimitedFiles
using InteractiveUtils
using MPI
using Random
using ..Output

import ..BiasModule: NoBias, calc_weights, recalc_CV!, update_bias!, write_to_file
import ..Gaugefields: calc_gauge_action, normalize!
import ..Measurements: MeasurementMethods, calc_measurements, calc_measurements_flowed
import ..Parameters: construct_params_from_toml
import ..Smearing: GradientFlow
import ..Universe: Univ
import ..Updates: HMC, ParityUpdate, Updatemethod, update!, temper!

export run_build, run_sim

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const myrank = MPI.Comm_rank(comm)
const comm_size = MPI.Comm_size(comm)

function print_acceptance_rates(numaccepts, itrj)
    myrank != 0 && return nothing

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
