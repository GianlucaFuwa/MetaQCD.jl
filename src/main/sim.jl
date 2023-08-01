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

function run_sim!(univ::Univ{TG, TB, TV}, parameters) where {TG, TB, TV}
    U = univ.U

    if univ.tempering_enabled
        updatemethod = Updatemethod(parameters, U[1])
        updatemethod_pt = HMCUpdate(
            U[1],
            parameters.hmc_integrator,
            parameters.hmc_steps,
            parameters.hmc_Δτ,
            meta_enabled = true,
        )
        parity = parameters.parity_update ? ParityUpdate(U[1]) : nothing
    else
        updatemethod = Updatemethod(parameters, U)
        parity = parameters.parity_update ? ParityUpdate(U) : nothing
    end

    println("\t>> Updatemethods are set!\n")

    if univ.tempering_enabled
        gradient_flow = GradientFlow(
            U[1],
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            tf = parameters.flow_tf,
            measure_every = parameters.flow_measure_every,
        )
        measurements = Vector{MeasurementMethods}(undef, parameters.numinstances)
        measurements[1] = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurement_methods,
            cv = parameters.meta_enabled,
            additional_string = "_0"
        )

        for i in 2:parameters.numinstances
            measurements[i] = MeasurementMethods(
                U[i],
                parameters.measuredir,
                parameters.measurement_methods,
                cv = parameters.meta_enabled,
                additional_string = "_$(i-1)"
            )
        end

        measurements_with_flow = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
        )
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
            cv = parameters.meta_enabled,
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
    Bias = univ.Bias
    vp = univ.verbose_print

    value, runtime_therm = @timed begin
        for itrj in 1:parameters.Ntherm
            println_verbose1(vp, "\n# therm itrj = $itrj")
            updatetime = 0.0

            _, updatetime = @timed begin
                update!(updatemethod, U, vp, Bias = Bias, metro_test = false)
            end

            println_verbose1(vp, ">> Therm. Update elapsed time:\t$(updatetime) [s]")
        end
    end

    println_verbose1(vp, "\t>> Thermalization elapsed time:\t$(runtime_therm) [s]")

    value, runtime_all = @timed begin
        numaccepts = 0.0
        for itrj in 1:parameters.Nsteps
            println_verbose1(vp, "\n# itrj = $itrj")

            _, updatetime = @timed begin
                numaccepts += update!(updatemethod, U, vp, Bias = Bias, metro_test = true)
                rand() < 0.5 ? update!(parity, U) : nothing
                update_bias!(Bias, U.CV)
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

            calc_weights(Bias, U.CV, itrj)
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
    Bias = univ.Bias
    vp = univ.verbose_print
    swap_every = parameters.swap_every

    value, runtime_therm = @timed begin
        for itrj in 1:parameters.Ntherm
            println_verbose1(vp, "\n# therm itrj = $itrj")
            updatetime = 0.0

            _, updatetime = @timed begin
                update!(updatemethod, U[1], vp, Bias = Bias[1], metro_test = false)

                for i in 2:numinstances
                    update!(updatemethod_pt, U[i], vp, Bias = Bias[i], metro_test = false)
                end
            end

            println_verbose1(vp, ">> Thermalization Update: Elapsed time $(updatetime) [s]")
        end
    end

    println_verbose1(vp, "\t>> Thermalization Elapsed time $(runtime_therm) [s]")

    value, runtime_all = @timed begin
        numaccepts = zeros(numinstances)
        numaccepts_temper = numinstances > 1 ? zeros(Int64, numinstances - 1) : nothing

        for itrj in 1:parameters.Nsteps
            println_verbose1(vp, "\n# itrj = $itrj")

            _, updatetime = @timed begin
                numaccepts[1] += update!(
                    updatemethod,
                    U[1],
                    vp,
                    Bias = Bias[1],
                    metro_test = true,
                )
                rand() < 0.5 ? update!(parity, U[1]) : nothing

                for i in 2:numinstances
                    numaccepts[i] += update!(
                        updatemethod_pt,
                        U[i],
                        vp,
                        Bias = Bias[i],
                        metro_test = true,
                    )
                    update_bias!(Bias[i], U[i].CV)
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
                    recalc_CV!(U[1], Bias[1])
                end

                for i in numinstances:-1:2
                    accepted = temper!(U[i], U[i-1], Bias[i], Bias[i-1], vp)
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
                measurestrings = calc_measurements(measurements[i], itrj, U[i])

                for value in measurestrings
                    println(value)
                end

                if i == 1
                    measurestrings_flowed = calc_measurements_flowed(
                        measurements_with_flow,
                        gradient_flow,
                        itrj,
                        U[1],
                    )
                    for value in measurestrings_flowed
                        println(value)
                    end
                end

                calc_weights(Bias[i], U[i].CV, itrj)
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
