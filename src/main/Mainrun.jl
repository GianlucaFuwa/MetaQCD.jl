module Mainrun
    using Dates
    using DelimitedFiles
    using InteractiveUtils
    using MPI
    using Random
    using ..Output

    import ..Gaugefields: normalize!
    import ..Measurements: MeasurementMethods, calc_measurements, calc_measurements_flowed
    import ..Metadynamics: calc_weights, recalc_CV!, update_bias!
    import ..Parameters: construct_params_from_toml
    import ..Smearing: GradientFlow
    import ..Universe: Univ
    import ..Updates: HMCUpdate, ParityUpdate, Updatemethod, update!, temper!

    export run_build, run_sim

    """
    So we don't have to type "if myrank == 0" all the time...
    """
    function println_rank0(val...)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println(v, val...)
        end
    end

    function println_verbose0(v::T, val...) where {T <: VerboseLevel}
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println_verbose1(v, val...)
        end
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

    include("build.jl")
    include("sim.jl")

end
