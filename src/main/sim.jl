function run_sim(filenamein::String)
    filename_head = splitext(filenamein)[1]
    filename = filename_head * ".toml"
    parameters = construct_params_from_toml(filename)

    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed)
    else
        seed = rand(1:1_000_000_000)
        Random.seed!(seed)
    end

    univ = Univ(parameters)
    println_verbose1(univ.verbose_print, "# ", pwd())
    println_verbose1(univ.verbose_print, "# ", Dates.now())
    io = IOBuffer()

    InteractiveUtils.versioninfo(io)
    versioninfo = String(take!(io))
    println_verbose1(univ.verbose_print, versioninfo)
    println_verbose1(univ.verbose_print, ">> Random seed is: $seed")
    println("\t>> Universe is set!\n")

    run_sim!(univ, parameters)

    return nothing
end

function run_sim!(univ, parameters)
    U = univ.U

    if parameters.tempering_enabled
        updatemethod = Updatemethod(parameters, U[1])
        updatemethod_pt = HMCUpdate(
            U[1],
            parameters.hmc_integrator,
            parameters.hmc_steps,
            parameters.hmc_deltatau,
            parameters.hmc_friction,
            bias_enabled = true,
        )
        parity = parameters.parity_update ? ParityUpdate(U[1]) : nothing
    else
        updatemethod = Updatemethod(parameters, U)
        parity = parameters.parity_update ? ParityUpdate(U) : nothing
    end

    println("\t>> Updatemethods are set!\n")

    if parameters.tempering_enabled
        gradient_flow = GradientFlow(
            U[1],
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            tf = parameters.flow_tf,
            measure_every = parameters.flow_measure_every,
        )
        measurements = Vector{MeasurementMethods}(undef, parameters.numinstances)
        measurements_with_flow = Vector{MeasurementMethods}(undef, parameters.numinstances)
        measurements[1] = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurement_methods,
            cv = true,
            additional_string = "_1",
        )
        measurements_with_flow[1] = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
            additional_string = "_1",
        )
        for i in 2:parameters.numinstances
            if parameters.measure_on_all
                measurements[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    parameters.measurement_methods,
                    cv = true,
                    additional_string = "_$i",
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    parameters.measurements_with_flow,
                    flow = true,
                    additional_string = "_$i",
                )
            else
                measurements[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    Dict[],
                    cv = true,
                    additional_string = "_$i",
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    Dict[],
                    additional_string = "_$i",
                )
            end
        end
    else
        gradient_flow = GradientFlow(
            U,
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            tf = parameters.flow_tf,
            measure_every = parameters.flow_measure_every,
        )
        measurements = MeasurementMethods(
            U,
            parameters.measuredir,
            parameters.measurement_methods,
            cv = parameters.kind_of_bias!="none",
        )
        measurements_with_flow = MeasurementMethods(
            U,
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
        )
    end

    println("\t>> Measurement methods are set!\n")

    save_configs = SaveConfigs(
        parameters.saveU_format,
        parameters.saveU_dir,
        parameters.saveU_every,
        univ.verbose_print
    )

    if parameters.tempering_enabled
        metaqcd_PT!(
            parameters,
            univ,
            updatemethod,
            updatemethod_pt,
            gradient_flow,
            measurements,
            measurements_with_flow,
            parity,
            save_configs,
        )
    else
        metaqcd!(
            parameters,
            univ,
            updatemethod,
            gradient_flow,
            measurements,
            measurements_with_flow,
            parity,
            save_configs,
        )
    end

    return nothing
end

function metaqcd!(
    parameters,
    univ,
    updatemethod,
    gradient_flow,
    measurements,
    measurements_with_flow,
    parity,
    save_configs,
)
    U = univ.U
    bias = univ.bias
    vp = univ.verbose_print

    value, runtime_therm = @timed begin
        for itrj in 1:parameters.numtherm
            println_verbose1(vp, "\n# therm itrj = $itrj")
            updatetime = 0.0

            _, updatetime = @timed begin
                update!(updatemethod, U, vp, bias=nothing, metro_test=false)
            end

            println_verbose1(vp, ">> Therm. Update elapsed time:\t$(updatetime) [s]")
        end
    end

    println_verbose1(vp, "\t>> Thermalization elapsed time:\t$(runtime_therm) [s]")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    value, runtime_all = @timed begin
        numaccepts = 0.0
        for itrj in 1:parameters.numsteps
            println_verbose1(vp, "\n# itrj = $itrj")

            _, updatetime = @timed begin
                numaccepts += update!(updatemethod, U, vp, bias=bias, metro_test=true)
                rand() < 0.5 ? update!(parity, U) : nothing
                update_bias!(bias, U.CV, itrj, true)
            end

            println_verbose1(vp, ">> Acceptance $itrj:\t", numaccepts * 100 / itrj, "%")
            println_verbose1(vp, ">> Update elapsed time:\t$(updatetime) [s]")

            save_gaugefield(save_configs, U, itrj)

            measurestrings, meas_time = @timed calc_measurements(measurements, itrj, U)
            measurestrings_flowed, flowmeas_time = @timed calc_measurements_flowed(
                measurements_with_flow,
                gradient_flow,
                itrj,
                U,
            )

            println_verbose1(
                vp,
                ">> Meas. elapsed time:\t$(meas_time) [s]\n",
                ">> FMeas. elapsed time:\t$(flowmeas_time) [s]",
            )

            for value in measurestrings
                println(value)
            end

            for value in measurestrings_flowed
                println(value)
            end

            calc_weights(bias, U.CV, itrj)
            flush(vp.fp)
        end
    end

    println_verbose1(vp, "\n\t>> Total elapsed time:\t$(convert_seconds(runtime_all)) \n")
    flush(stdout)
    flush(vp)
    # close(measurements)
    # close(measurements_with_flow)
    return nothing
end

function metaqcd_PT!(
    parameters,
    univ,
    updatemethod,
    updatemethod_pt,
    gradient_flow,
    measurements,
    measurements_with_flow,
    parity,
    save_configs,
)
    numinstances = parameters.numinstances
    U = univ.U
    bias = univ.bias
    vp = univ.verbose_print
    swap_every = parameters.swap_every
    rank0_updates = parameters.non_metadynamics_updates
    measure_on_all = parameters.measure_on_all

    value, runtime_therm = @timed begin
        for itrj in 1:parameters.numtherm
            println_verbose1(vp, "\n# therm itrj = $itrj")
            updatetime = 0.0
            # thermalize without bias potential contribution, since it's a waste of time
            _, updatetime = @timed begin
                update!(updatemethod, U[1], vp, bias=nothing, metro_test=false)

                for i in 2:numinstances
                    update!(updatemethod_pt, U[i], vp, bias=nothing, metro_test=false)
                end
            end

            println_verbose1(vp, ">> Thermalization Update: Elapsed time $(updatetime) [s]")
        end
    end

    println_verbose1(vp, "\t>> Thermalization Elapsed time $(runtime_therm) [s]")
    for i in 2:numinstances
        recalc_CV!(U[i], bias[i]) # need to recalc cv since it was not updated during therm
    end

    value, runtime_all = @timed begin
        numaccepts = zeros(numinstances)
        numaccepts_temper = zeros(Int64, numinstances-1)

        for itrj in 1:parameters.numsteps
            println_verbose1(vp, "\n# itrj = $itrj")
            _, updatetime = @timed begin
                tmp = 0.0
                for _ in 1:rank0_updates
                    tmp += update!(updatemethod, U[1], vp, bias=nothing)
                end
                numaccepts[1] += tmp/rank0_updates
                rand() < 0.5 ? update!(parity, U[1]) : nothing

                for i in 2:numinstances
                    numaccepts[i] += update!(updatemethod_pt, U[i], vp, bias=bias[i])
                    update_bias!(bias[i], U[i].CV, itrj, true)
                end
            end

            println_verbose1(vp, ">> Update: Elapsed time $(sum(updatetime)) [s]")

            for (i, value) in enumerate(numaccepts)
                println_verbose1(vp, ">> Acceptance $i $itrj:\t$(value * 100 / itrj) %")
            end

            if itrj % swap_every == 0
                # We need to recalculate the CV and for the non-MetaD stream since that is
                # only done in MetaD-HMC updates
                if (typeof(updatemethod) <: HMCUpdate) == false
                    recalc_CV!(U[1], bias[1])
                end

                for i in numinstances:-1:2
                    accepted = temper!(U[i], U[i-1], bias[i], bias[i-1], itrj, vp)
                    numaccepts_temper[i-1] += accepted

                    println_verbose1(
                        vp,
                        ">> Swap Acceptance [$i ⇔  $(i-1)] $itrj:\t",
                        "$(numaccepts_temper[i-1] * 100 / (itrj / swap_every)) %"
                    )
                end
            end

            save_gaugefield(save_configs, U[1], itrj)

            for i in 1:numinstances
                measurestrings = calc_measurements(measurements[i], itrj, U[i]; str="$i")

                for value in measurestrings
                    println(value)
                end

                if i == 1 || measure_on_all
                    measurestrings_flowed = calc_measurements_flowed(
                        measurements_with_flow[i],
                        gradient_flow,
                        itrj,
                        U[i],
                        str = "$i"
                    )
                    for value in measurestrings_flowed
                        println(value)
                    end
                end

                calc_weights(bias[i], U[i].CV, itrj)
            end

            flush(vp.fp)
        end
    end

    println_verbose1(vp, "\n\t>> Total elapsed time:\t$(convert_seconds(runtime_all)) \n")
    flush(stdout)
    flush(vp)
    # close(measurements)
    # close(measurements_with_flow)
    return nothing
end
