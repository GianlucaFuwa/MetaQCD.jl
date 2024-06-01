struct MeasurementMethods
    measurement_parameters_set::Vector{MeasurementParameters} # Parameters of each observable
    measurements::Vector{AbstractMeasurement} # Vector of obeservables to be measured
    num_measurements::Int64 # number of measurements
    intervals::Vector{Int64} # measure observable[i] every intervals[i] updates
end

function MeasurementMethods(
    U, measurement_dir, measurement_methods::Vector{Dict}; flow=false, additional_string=""
)
    @level1("┌ Preparing $(ifelse(flow, "flowed", "")) Measurements...")
    nummeasurements = length(measurement_methods)
    measurements = Vector{AbstractMeasurement}(undef, nummeasurements)
    measurement_parameters_set = Vector{MeasurementParameters}(undef, nummeasurements)
    intervals = zeros(Int64, nummeasurements)

    str = flow ? "_flowed" : ""

    for (i, method) in enumerate(measurement_methods)
        measurement_parameters = construct_measurement_parameters_from_dict(method)
        @level1("|  OBSERVABLE $i: $(measurement_parameters.methodname)")
        intervals[i] = measurement_parameters.measure_every
        @level1("|    every $(intervals[i]) updates")
        if measurement_parameters.methodname == "wilson_loop"
            T = measurement_parameters.Tmax 
            R = measurement_parameters.Rmax 
            @level1("|    Maximum Extends: $T x $R (only even extends are measured, sorry, will fix soon)")
            @level1("|    @info: Wilson loop measurements are not printed to console")
        end
        if measurement_parameters.methodname == "pion_correlator"
            dirac_type = measurement_parameters.dirac_type
            mass = measurement_parameters.mass
            csw = dirac_type == "wilson" ? measurement_parameters.csw : nothing
            eo_precon = measurement_parameters.eo_precon
            cg_tol = measurement_parameters.cg_tol
            cg_maxiters = measurement_parameters.cg_maxiters
            @level1("|    Dirac Operator: $(dirac_type)")
            @level1("|    Mass: $(mass)")
            dirac_type == "wilson" && @level1("|    CSW: $(csw)")
            @level1("|    Even-odd preconditioned: $(eo_precon)")
            @level1("|    CG Tolerance: $(cg_tol)")
            @level1("|    CG Max Iterations: $(cg_maxiters)")
        end
        filename =
            measurement_dir *
            "/" *
            measurement_parameters.methodname *
            additional_string *
            "$str.txt"
        measurements[i] = prepare_measurement(U, measurement_parameters, filename, flow)
        measurement_parameters_set[i] = deepcopy(measurement_parameters)
    end

    @level1("└\n")
    return MeasurementMethods(
        measurement_parameters_set, measurements, nummeasurements, intervals
    )
end

function calc_measurements(m::Vector{MeasurementMethods}, U, itrj)
    for i in eachindex(m)
        calc_measurements(m[i], U[i], itrj; str="$i")
    end
    return nothing
end

function calc_measurements(m::MeasurementMethods, U, itrj; str="")
    # check if the current iteration has any measurements to be made to avoid work
    check_for_measurements(itrj, m.intervals) || return nothing

    for i in 1:(m.num_measurements)
        interval = m.intervals[i]

        if itrj % interval == 0
            out = measure(m.measurements[i], U; additional_string="$itrj")
            @level1(get_string(out) * str)
        end
    end

    return nothing
end

function calc_measurements_flowed(
    m::Vector{MeasurementMethods}, gflow, U, itrj, measure_on_all=false
)
    if measure_on_all # if we measure on all streams in PT-MetaD
        for i in eachindex(m)
            calc_measurements_flowed(m[i], gflow, U[i], itrj; str="$i")
        end
    else
        calc_measurements_flowed(m[1], gflow, U[1], itrj; str="1")
    end

    return nothing
end

function calc_measurements_flowed(m::MeasurementMethods, gradient_flow, U, itrj; str="")
    # check if the current iteration has any measurements to be made to avoid work
    check_for_measurements(itrj, m.intervals) || return nothing
    copy!(gradient_flow.Uflow, U)
    tf = gradient_flow.tf

    for iflow in 1:(gradient_flow.numflow)
        τ = round(iflow * tf; sigdigits=3)
        flow!(gradient_flow)

        if iflow ∈ gradient_flow.measure_at
            additional_string = @sprintf("%-9i\t%-7i\t%-9.5f", itrj, iflow, τ)

            for i in 1:(m.num_measurements)
                interval = m.intervals[i]

                if itrj % interval == 0
                    out = measure(
                        m.measurements[i],
                        gradient_flow.Uflow;
                        additional_string=additional_string,
                    )
                    @level1(get_string(out) * str)
                end
            end
        end
    end

    return nothing
end

function check_for_measurements(itrj, intervals)
    for num in intervals
        (itrj % num == 0) && return true
    end

    return false
end

function Base.close(m::MeasurementMethods)
    for meas in m.measurements
        close(meas)
    end

    return nothing
end
