struct MeasurementMethods{M}
    # measurement_parameters_set::Vector{MeasurementParameters} # Parameters of each observable
    measurements::M # Vector of obeservables to be measured
    intervals::Vector{Int64} # measure observable[i] every intervals[i] updates
    num_measurements::Int64 # number of measurements
end

@inline Base.getindex(m::MeasurementMethods{M}, i) where {M} = m.measurements[i]
@inline Base.eachindex(m::MeasurementMethods{M}) where {M} = Base.OneTo(length(m.measurements))

function MeasurementMethods(
    U, measurement_dir, measurement_methods::Vector{Dict}; flow=false, additional_string=""
)
    @level1("┌ Preparing $(ifelse(flow, "flowed", "")) Measurements...")
    num_measurements = length(measurement_methods)
    # measurement_parameters_set = Vector{MeasurementParameters}(undef, num_measurements)
    intervals = zeros(Int64, num_measurements)

    str = flow ? "_flowed" : ""

    measurements = ntuple(num_measurements) do i
        measurement_parameters = meas_parameters_from_dict(measurement_methods[i])
        name = measurement_parameters.methodname
        @level1("|  OBSERVABLE $i: $(measurement_parameters.methodname)")
        intervals[i] = measurement_parameters.measure_every
        @level1("|    every $(intervals[i]) updates")
        filename = joinpath(measurement_dir, name * additional_string * "$str")
        # measurement_parameters_set[i] = deepcopy(measurement_parameters)
        prepare_measurement(U, measurement_parameters, filename, flow)
    end

    @level1("└\n")
    return MeasurementMethods(measurements, intervals, num_measurements)
end

function calc_measurements(m::Vector{MeasurementMethods}, U, itrj, measure_on_all=false)
    if measure_on_all # if we measure on all streams in PT-MetaD
        for i in eachindex(m)
            calc_measurements(m[i], U[i], itrj, i-1)
        end
    else
        calc_measurements(m[1], U[1], itrj, 0)
    end

    return nothing
end

function calc_measurements(m::MeasurementMethods, U, itrj, myinstance=MYRANK)
    # check if the current iteration has any measurements to be made to avoid work
    check_for_measurements(itrj, m.intervals) || return nothing

    for i in eachindex(m)
        interval = m.intervals[i]

        if itrj%interval == 0
            measure(m[i], U, myinstance, itrj)
        end
    end

    return nothing
end

function calc_measurements_flowed(
    m::Vector{MeasurementMethods}, gflow, U, itrj, measure_on_all=false
)
    if measure_on_all # if we measure on all streams in PT-MetaD
        for i in eachindex(m)
            calc_measurements_flowed(m[i], gflow, U[i], itrj, i-1)
        end
    else
        calc_measurements_flowed(m[1], gflow, U[1], itrj, 0)
    end

    return nothing
end

function calc_measurements_flowed(
    m::MeasurementMethods, gradient_flow, U, itrj, myinstance=MYRANK
)
    # check if the current iteration has any measurements to be made to avoid work
    check_for_measurements(itrj, m.intervals) || return nothing
    copy!(gradient_flow.Uflow, U)
    tf = gradient_flow.tf

    for iflow in 1:(gradient_flow.numflow)
        flow!(gradient_flow)
        Uflow = gradient_flow.Uflow
        τ = iflow * tf

        if iflow ∈ gradient_flow.measure_at
            for i in 1:(m.num_measurements)
                interval = m.intervals[i]

                if itrj%interval == 0
                    measure(m.measurements[i], Uflow, myinstance, itrj, (iflow, τ))
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
