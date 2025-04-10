struct MeasurementMethods{M}
    # measurement_parameters_set::Vector{MeasurementParameters} # Parameters of each observable
    measurements::M # Vector of obeservables to be measured
    intervals::Vector{Int64} # measure observable[i] every intervals[i] updates
    num_measurements::Int64 # number of measurements
end

@inline Base.getindex(m::MeasurementMethods{M}, i) where {M} = m.measurements[i]
@inline Base.eachindex(m::MeasurementMethods{M}) where {M} = Base.OneTo(length(m.measurements))
@inline flow_string(::Nothing) = ""
@inline flow_string(::NoSmearing) = ""
@inline flow_string(::GradientFlow) = "_gflow"
@inline flow_string(::Cooling) = "_cooling"

function MeasurementMethods(
    U, measurement_dir, measurement_methods::Vector{Dict};
    flow=NoSmearing(), additional_string="",
)
    fstr = filter(x -> x!='_', flow_string(flow))
    @level1("- Preparing$(fstr) Measurements...")
    num_measurements = length(measurement_methods)
    # measurement_parameters_set = Vector{MeasurementParameters}(undef, num_measurements)
    intervals = zeros(Int64, num_measurements)

    add_string = flow_string(flow) * additional_string

    measurements = ntuple(num_measurements) do i
        measurement_parameters = meas_parameters_from_dict(measurement_methods[i])
        name = measurement_parameters.methodname
        @level1("|  OBSERVABLE $i: $(name)")
        intervals[i] = measurement_parameters.measure_every
        @level1("|    interval: $(intervals[i])")
        filename = joinpath(measurement_dir, name * add_string)
        # measurement_parameters_set[i] = deepcopy(measurement_parameters)
        prepare_measurement(U, measurement_parameters, filename, flow)
    end

    @level1("-\n")
    return MeasurementMethods(measurements, intervals, num_measurements)
end

function calc_measurements(
    m::Vector{MeasurementMethods}, U, itrj, measure_on_all=false; kwargs...
)
    if measure_on_all # if we measure on all streams in PT-MetaD
        for i in eachindex(m)
            calc_measurements(m[i], U[i], itrj)
        end
    else
        calc_measurements(m[1], U[1], itrj)
    end

    return nothing
end

function calc_measurements(m::MeasurementMethods, U, itrj; mpi_multi_sim=false)
    # check if the current iteration has any measurements to be made to avoid work
    check_for_measurements(itrj, m.intervals) || return nothing

    for i in eachindex(m)
        interval = m.intervals[i]

        if itrj%interval == 0
            measure(m[i], U, itrj, nothing; mpi_multi_sim=mpi_multi_sim)
        end
    end

    return nothing
end

calc_measurements_flowed(::Any, ::NoSmearing, args...; kwargs...) = nothing

function calc_measurements_flowed(
    m::Vector{MeasurementMethods}, flow, U, itrj, measure_on_all=false; kwargs...
)
    if measure_on_all # if we measure on all streams in PT-MetaD
        for i in eachindex(m)
            calc_measurements_flowed(m[i], flow, U[i], itrj)
        end
    else
        calc_measurements_flowed(m[1], flow, U[1], itrj)
    end

    return nothing
end


function calc_measurements_flowed(m::Tuple, flow::Tuple, U, itrj; mpi_multi_sim=false)
    for i in eachindex(flow)
        calc_measurements_flowed(m[i], flow[i], U, itrj; mpi_multi_sim=mpi_multi_sim)
    end

    return nothing
end

function calc_measurements_flowed(
    m::MeasurementMethods, flow::AbstractSmearing, U, itrj; mpi_multi_sim=false
)
    # check if the current iteration has any measurements to be made to avoid work
    check_for_measurements(itrj, m.intervals) || return nothing
    copy!(flow.Uflow, U)
    tf = flow.tf

    for iflow in 1:(flow.numflow)
        flow!(flow)
        Uflow = flow.Uflow
        τ = iflow * tf

        if iflow ∈ flow.measure_at
            for i in 1:(m.num_measurements)
                interval = m.intervals[i]

                if itrj%interval == 0
                    measure(
                        m.measurements[i], Uflow, itrj, (iflow, τ);
                        mpi_multi_sim=mpi_multi_sim, fstr=flow_string(flow)
                    )
                end
            end
        end
    end

    return nothing
end

@inline function check_for_measurements(itrj, intervals)
    for num in intervals
        (itrj%num == 0) && return true
    end

    return false
end
