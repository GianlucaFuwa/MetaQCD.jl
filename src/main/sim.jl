function run_sim(filenamein::String)
    filename_head = splitext(filenamein)[1]
    filename = filename_head * ".toml"
    parameters = construct_params_from_toml(filename)

    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed)
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    univ = Univ(parameters)

    println_verbose1(univ.verbose_print, ">> Random seed is: $seed\n")

    run_sim!(univ, parameters)

    return nothing
end

function run_sim!(univ, parameters)
    U = univ.U

    if parameters.tempering_enabled
        updatemethod = Updatemethod(parameters, U[1], univ.verbose_print)
        updatemethod_pt = HMC(
            U[1],
            parameters.hmc_integrator,
            parameters.hmc_steps,
            parameters.hmc_trajectory;
            friction = parameters.hmc_friction,
            bias_enabled = true,
            verbose = univ.verbose_print,
        )
        parity = parameters.parity_update ? ParityUpdate(U[1]) : nothing
    else
        updatemethod = Updatemethod(parameters, U, univ.verbose_print)
        parity = parameters.parity_update ? ParityUpdate(U) : nothing
    end
    parity≢nothing && println_verbose1(univ.verbose_print, "\t>> PARITY UPDATE ENABLED\n")

    if parameters.tempering_enabled
        gflow = GradientFlow(
            U[1],
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            tf = parameters.flow_tf,
            measure_every = parameters.flow_measure_every,
            verbose = univ.verbose_print,
        )
        measurements = Vector{MeasurementMethods}(undef, parameters.numinstances)
        measurements_with_flow = Vector{MeasurementMethods}(undef, parameters.numinstances)
        println_verbose1(univ.verbose_print, ">> Preparing Measurements...")
        measurements[1] = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurements,
            additional_string = "_0",
            verbose = univ.verbose_print,
        )
        println_verbose1(univ.verbose_print, ">> Preparing flowed Measurements...")
        measurements_with_flow[1] = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
            additional_string = "_0",
            verbose = univ.verbose_print,
        )
        for i in 2:parameters.numinstances
            if parameters.measure_on_all
                measurements[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    parameters.measurements,
                    additional_string = "_$(i-1)",
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    parameters.measurements_with_flow,
                    flow = true,
                    additional_string = "_$(i-1)",
                )
            else
                measurements[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    Dict[],
                    additional_string = "_$(i-1)",
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    Dict[],
                    additional_string = "_$(i-1)",
                )
            end
        end
    else
        gflow = GradientFlow(
            U,
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            tf = parameters.flow_tf,
            measure_every = parameters.flow_measure_every,
            verbose = univ.verbose_print,
        )
        println_verbose1(univ.verbose_print, ">> Preparing Measurements...")
        measurements = MeasurementMethods(
            U,
            parameters.measuredir,
            parameters.measurements,
            verbose = univ.verbose_print,
        )
        println_verbose1(univ.verbose_print, ">> Preparing flowed Measurements...")
        measurements_with_flow = MeasurementMethods(
            U,
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
            verbose = univ.verbose_print,
        )
    end

    println(">> Measurement methods are set!\n")

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
            gflow,
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
            gflow,
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
    gflow,
    measurements,
    measurements_with_flow,
    parity,
    save_configs,
)
    U = univ.U
    bias = univ.bias
    vp = univ.verbose_print

    _, runtime_therm = @timed begin
        for itrj in 1:parameters.numtherm
            println_verbose1(vp, "\n# therm itrj = $itrj")
            updatetime = 0.0

            _, updatetime = @timed begin
                update!(updatemethod, U, vp, bias=nothing, metro_test=false)
            end

            println_verbose1(vp, ">> Therm. Update elapsed time:\t$(updatetime) [s]\n#")
        end
    end

    println_verbose1(vp, "\t>> Thermalization elapsed time:\t$(runtime_therm) [s]\n")
    # load in config and recalculate gauge action if given
    load_gaugefield!(U, parameters, vp) && (U.Sg = calc_gauge_action(U))
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    _, runtime_all = @timed begin
        numaccepts = 0.0
        for itrj in 1:parameters.numsteps
            println_verbose1(vp, "\n# itrj = $itrj")

            _, updatetime = @timed begin
                numaccepts += update!(updatemethod, U, vp, bias=bias, metro_test=true)
                rand()<0.5 ? update!(parity, U) : nothing
                update_bias!(bias, U.CV, itrj, true)
            end

            print_acceptance_rates(numaccepts, itrj, vp)
            println_verbose1(vp, ">> Update elapsed time:\t$(updatetime) [s]")

            save_gaugefield(save_configs, U, vp, itrj)

            _, mtime = @timed calc_measurements(measurements, U, itrj)
            _, fmtime = @timed calc_measurements_flowed(measurements_with_flow, gflow, U, itrj)
            calc_weights(bias, U.CV, itrj)
            println_verbose1(
                vp,
                ">> Meas. elapsed time:\t$(mtime) [s]\n",
                ">> FMeas. elapsed time:\t$(fmtime) [s]\n",
                "#",
            )
            flush(vp)
        end
    end

    println_verbose1(vp, "\n\t>> Total elapsed time:\t$(convert_seconds(runtime_all)) \n")
    flush(stdout)
    close(measurements)
    close(measurements_with_flow)
    close(bias)
    close(vp)
    return nothing
end

function metaqcd_PT!(
    parameters,
    univ,
    updatemethod,
    updatemethod_pt,
    gflow,
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

    # if stream 1 uses hmc then we have to recalc the CV before tempering
    uses_hmc = <:(typeof(updatemethod), HMC)

    value, runtime_therm = @timed begin
        for itrj in 1:parameters.numtherm
            println_verbose1(vp, "\n# therm itrj = $itrj")
            # thermalize without bias potential contribution, since it's a waste of time
            _, updatetime = @timed begin
                for _ in 1:rank0_updates
                    for i in 1:numinstances
                        update!(updatemethod, U[i], vp, bias=nothing, metro_test=!uses_hmc)
                    end
                end
            end

            println_verbose1(vp, ">> Thermalization Update: Elapsed time $(updatetime) [s]\n#")
        end
    end

    println_verbose1(vp, "\t>> Thermalization Elapsed time $(runtime_therm) [s]")
    # load in config and recalculate gauge action if given
    load_gaugefield!(U[1], parameters, vp) && (U[1].Sg = calc_gauge_action(U[1]))
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

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
                rand()<0.5 && update!(parity, U[1])

                for i in 2:numinstances
                    numaccepts[i] += update!(updatemethod_pt, U[i], vp, bias=bias[i])
                    update_bias!(bias[i], U[i].CV, itrj, true)
                end
            end

            println_verbose1(vp, ">> Update: Elapsed time $(sum(updatetime)) [s]")
            print_acceptance_rates(numaccepts, itrj, vp)

            temper!(U, bias, numaccepts_temper, swap_every, itrj, vp; recalc=!uses_hmc)

            save_gaugefield(save_configs, U[1], vp, itrj)

            _, mtime = @timed calc_measurements(measurements, U, itrj)
            _, fmtime = @timed calc_measurements_flowed(measurements_with_flow, gflow, U, itrj, measure_on_all)
            calc_weights(bias, [U[i].CV for i in 1:numinstances], itrj)
            println_verbose1(
                vp,
                ">> Meas. elapsed time:\t$(mtime) [s]\n",
                ">> FMeas. elapsed time:\t$(fmtime) [s]\n",
                "#",
            )
            flush(vp.fp)
        end
    end

    flush(stdout)
    [close(m)  for m in measurements]
    [close(mf) for mf in measurements_with_flow]
    [close(b)  for b in bias]
    close(vp)

    println_verbose1(vp, "\n\t>> Total elapsed time:\t$(convert_seconds(runtime_all)) \n")
    return nothing
end
