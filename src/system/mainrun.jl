module Mainrun
    using Dates
    using DelimitedFiles
    using InteractiveUtils
    using Random
    using ..IOModule
    using ..VerbosePrint

    import ..AbstractMeasurementModule: measure
    import ..AbstractSmearingModule: GradientFlow, Euler, RK2, RK3, RK3W7
    import ..AbstractUpdateModule: HeatbathUpdate, HMCUpdate, MetroUpdate, Updatemethod
    import ..AbstractUpdateModule: AbstractUpdate, ParityUpdate, update!, parity_update!
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTreeGaugeAction, SymanzikTadGaugeAction, WilsonGaugeAction
    import ..Metadynamics: MetaDisabled, MetaEnabled, calc_weights, update_bias!
    import ..MetaQCD: MeasurementMethods, calc_measurements, calc_measurements_flowed
    import ..ParametersTOML: construct_params_from_toml
    import ..UniverseModule: Univ
    import ..TemperingModule: temper!
    import ..VerbosePrint: VerboseLevel, println_verbose1

    function run_sim(filenamein::String)
        filename_head = splitext(filenamein)[1]
        filename = filename_head * ".toml"
        parameters = construct_params_from_toml(filename)

        Random.seed!(parameters.randomseed)

        univ = Univ(parameters)
        println_verbose1(univ.verbose_print, "# ", pwd())
        println_verbose1(univ.verbose_print, "# ", Dates.now())
        io = IOBuffer()

        InteractiveUtils.versioninfo(io)
        versioninfo = String(take!(io))
        println_verbose1(univ.verbose_print, versioninfo)

        println("\t>> Universe is set!\n")

        run_sim!(univ, parameters)

        return nothing
    end

    function run_sim!(univ, parameters)
        U = univ.U

        if parameters.tempering_enabled
            updatemethod = Vector{AbstractUpdate}(undef, univ.numinstances)
            updatemethod[1] = Updatemethod(parameters, U[1])

            for i in 2:univ.numinstances
                updatemethod[i] = HMCUpdate(
                    U[i],
                    parameters.hmc_integrator,
                    parameters.hmc_steps,
                    parameters.hmc_Δτ,
                    meta_enabled = true,
                )
            end
        else
            updatemethod = Updatemethod(parameters, U)
        end

        parity = parameters.parity_update ? ParityUpdate(U) : nothing

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
                    Dict[],
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

        if parameters.tempering_enabled
            metaqcd_PT!(
                parameters,
                univ,
                updatemethod,
                gradient_flow,
                measurements,
                measurements_with_flow,
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
    )
        save_configs = SaveConfigs(
            parameters.saveU_format,
            parameters.saveU_dir,
            parameters.saveU_every,
            univ.verbose_print
        )
        U = univ.U
        Bias = univ.Bias

        value, runtime_therm = @timed begin
            for itrj in 1:parameters.Ntherm
                println_verbose1(univ.verbose_print, "\n# therm itrj = $itrj")
                updatetime = 0.0

                _, updatetime = @timed update!(
                    updatemethod,
                    U,
                    univ.verbose_print,
                    Bias = Bias,
                    metro_test = false,
                )

                println_verbose1(
                    univ.verbose_print,
                    ">> Therm. Update elapsed time:\t$(updatetime) [s]"
                )
            end
        end

        println_verbose1(
            univ.verbose_print,
            "\t>> Thermalization elapsed time:\t$(runtime_therm) [s]"
        )

        value, runtime_all = @timed begin
            numaccepts = 0.0
            for itrj in 1:parameters.Nsteps
                println_verbose1(univ.verbose_print, "\n# itrj = $itrj")

                accepted, updatetime = @timed update!(
                    updatemethod,
                    U,
                    univ.verbose_print,
                    Bias = Bias,
                    metro_test = true,
                )

                if rand() < 0.5
                    parity_update!(U, parity)
                end

                update_bias!(Bias, U.CV)
                numaccepts += accepted

                println_verbose1(
                    univ.verbose_print,
                    ">> Acceptance $itrj:\t",
                    numaccepts * 100 / itrj,
                    "%",
                )

                println_verbose1(
                    univ.verbose_print,
                    ">> Update elapsed time:\t$(updatetime) [s]"
                )

                save_gaugefield(save_configs, U, itrj)

                measurestrings, meas_time = @timed calc_measurements(
                    measurements,
                    itrj,
                    U,
                )

                measurestrings_flowed, flowmeas_time = @timed calc_measurements_flowed(
                    measurements_with_flow,
                    gradient_flow,
                    itrj,
                    U,
                )

                println_verbose1(
                    univ.verbose_print,
                    ">> Meas. elapsed time:\t$(meas_time) [s]\n",
                    ">> FMeas. elapsed time:\t$(flowmeas_time) [s]",
                )

                for (i, value) in enumerate(measurestrings)
                    println(value)
                end

                for (i, value) in enumerate(measurestrings_flowed)
                    println(value)
                end

                calc_weights(Bias, U.CV, itrj)

                flush(univ.verbose_print.fp)
            end
        end

        if parameters.meta_enabled
            writedlm(
                Bias.fp,
                [Bias.bin_vals Bias.values],
            )

            close(Bias.fp)

            println_verbose1(
                univ.verbose_print,
                "\t>> Metapotential has been saved in file \"$(Bias.fp)\""
            )
        end

        println_verbose1(
            univ.verbose_print,
            "\n\t>> Total elapsed time:\t$(convert_from_seconds(runtime_all)) \n",
        )
        flush(stdout)
        flush(univ.verbose_print)
        # close(measurements)
        # close(measurements_with_flow)
        return nothing
    end

    function metaqcd_PT!(
        parameters,
        univ,
        updatemethod,
        gradient_flow,
        measurements,
        measurements_with_flow,
    )
        numinstances = parameters.numinstances
        save_configs = SaveConfigs(
            parameters.saveU_format,
            parameters.saveU_dir,
            parameters.saveU_every,
            univ.verbose_print,
        )

        U = univ.U
        Bias = univ.Bias

        value, runtime_therm = @timed begin
            for itrj in 1:parameters.Ntherm
                println_verbose1(univ.verbose_print, "\n# therm itrj = $itrj")
                updatetime = 0.0

                _, updatetime = @timed begin
                    for i in 1:numinstances
                        update!(
                            updatemethod[i],
                            U[i],
                            univ.verbose_print,
                            Bias = Bias[i],
                            metro_test = false,
                        )
                    end
                end

                println_verbose1(
                    univ.verbose_print,
                    ">> Thermalization Update: Elapsed time $(updatetime) [s]"
                )
            end
        end

        println_verbose1(
            univ.verbose_print,
            "\t>> Thermalization Elapsed time $(runtime_therm) [s]"
        )

        value, runtime_all = @timed begin
            numaccepts = zeros(numinstances)
            numaccepts_temper = numinstances > 1 ? zeros(Int64, numinstances - 1) : nothing

            for itrj in 1:parameters.Nsteps
                println_verbose1(univ.verbose_print, "\n# itrj = $itrj")

                _, updatetime = @timed begin
                    for i in 1:numinstances
                        accepted = update!(
                            updatemethod[i],
                            U[i],
                            univ.verbose_print,
                            Bias = Bias[i],
                            metro_test = true,
                        )
                        Bias[i] !== nothing ? update_bias!(Bias[i], U[i].CV) : nothing
                        numaccepts[i] += accepted
                    end
                end

                println_verbose1(
                    univ.verbose_print,
                    ">> Update: Elapsed time $(sum(updatetime)) [s]"
                )

                for (i, value) in enumerate(numaccepts)
                    println_verbose1(
                        univ.verbose_print,
                        ">> Acceptance rank_$i $itrj:\t",
                        value * 100 / itrj,
                        "%",
                    )
                end

                swap_every = parameters.swap_every

                if itrj % swap_every == 0
                    for i in numinstances:-1:2
                        accepted = temper!(
                            U[i],
                            U[i-1],
                            Bias[i],
                            Bias[i-1],
                            univ.verbose_print
                        )
                        numaccepts_temper[i-1] += accepted

                        println_verbose1(
                            univ.verbose_print,
                            ">> Swap Acceptance [$i ⇔  $(i-1)] $itrj:\t",
                            "$(numaccepts_temper[i-1] * 100 / (itrj / swap_every)) %"
                        )
                    end
                end

                save_gaugefield(save_configs, U[1], itrj)

                for i in 1:numinstances
                    measurestrings = calc_measurements(
                        measurements[i],
                        itrj,
                        U[i],
                    )
                    for (i, value) in enumerate(measurestrings)
                        println(value)
                    end

                    if i == 1
                        measurestrings_flowed = calc_measurements_flowed(
                            measurements_with_flow,
                            gradient_flow,
                            itrj,
                            U[1],
                        )
                        for (i, value) in enumerate(measurestrings_flowed)
                            println(value)
                        end
                    end

                    calc_weights(Bias[i], U[i].CV, itrj)
                end

                flush(univ.verbose_print.fp)
            end
        end

        if parameters.meta_enabled
            for i in 1:numinstances
                writedlm(
                    Bias[i].fp,
                    [Bias[i].bin_vals Bias[i].values],
                )

                close(Bias[i].fp)

                println_verbose1(
                    univ.verbose_print,
                    "\t>> Metapotential $i has been saved in file \"$(Bias[i].fp)\""
                )
            end
        end

        println_verbose1(univ.verbose_print, "\t>> Total Elapsed time $(runtime_all) [s]\n")
        flush(stdout)
        flush(univ.verbose_print)
        # close(measurements)
        # close(measurements_with_flow)
        return nothing
    end

    function convert_from_seconds(sec)
        sec = round(Int, sec, RoundNearestTiesAway)
        x, seconds = divrem(sec, 60)
        y, minutes = divrem(x, 60)
        days, hours = divrem(y, 24)
        return "$(Day(days)), $(Hour(hours)), $(Minute(minutes)), $(Second(seconds))"
    end

end
