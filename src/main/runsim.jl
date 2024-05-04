function run_sim(filenamein::String)
    filename_head = splitext(filenamein)[1]
    filename = filename_head * ".toml"

    # load parameters from toml file
    parameters = construct_params_from_toml(filename)

    # set random seed if provided, otherwise generate one
    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed)
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    set_global_logger!(
        parameters.verboselevel,
        parameters.logdir * "/logs.txt";
        tc=parameters.log_to_console,
    )

    # print time and system info, because it looks cool I guess
    # btw, all these "@level1" calls are just for logging, level1 is always printed
    # and anything higher has to specified in the parameter file (default is level2)
    @level1("# $(pwd()) @ $(current_time())")
    buf = IOBuffer()
    InteractiveUtils.versioninfo(buf)
    versioninfo = String(take!(buf))
    @level1(versioninfo)
    @level1("[ Random seed is: $seed\n")

    # Univ is a struct that holds the gaugefield and bias
    univ = Univ(parameters)

    run_sim!(univ, parameters)
    return nothing
end

function run_sim!(univ, parameters)
    U = univ.U

    # initialize update method, measurements, and bias
    if parameters.tempering_enabled
        updatemethod = Updatemethod(parameters, U[1])
        # all MetaD streams use HMC, so there is no need to initialize more than 1
        updatemethod_pt = HMC(
            U[1],
            parameters.hmc_integrator,
            parameters.hmc_trajectory,
            parameters.hmc_steps,
            parameters.hmc_friction;
            fermion_action=parameters.fermion_action,
            bias_enabled=true,
        )
        parity = parameters.parity_update ? ParityUpdate(U[1]) : nothing
    else
        updatemethod = Updatemethod(parameters, U)
        parity = parameters.parity_update ? ParityUpdate(U) : nothing
    end
    parity ≢ nothing && @level1("[ Parity update enabled\n")

    if parameters.tempering_enabled
        gflow = GradientFlow(
            U[1],
            parameters.flow_integrator,
            parameters.flow_num,
            parameters.flow_steps,
            parameters.flow_tf;
            measure_every=parameters.flow_measure_every,
        )
        measurements = Vector{MeasurementMethods}(undef, parameters.numinstances)
        measurements_with_flow = Vector{MeasurementMethods}(undef, parameters.numinstances)

        measurements[1] = MeasurementMethods(
            U[1], parameters.measuredir, parameters.measurements; additional_string="_0"
        )

        measurements_with_flow[1] = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurements_with_flow;
            flow=true,
            additional_string="_0",
        )
        for i in 2:(parameters.numinstances)
            if parameters.measure_on_all
                measurements[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    parameters.measurements;
                    additional_string="_$(i-1)",
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i],
                    parameters.measuredir,
                    parameters.measurements_with_flow;
                    flow=true,
                    additional_string="_$(i-1)",
                )
            else
                measurements[i] = MeasurementMethods(
                    U[i], parameters.measuredir, Dict[]; additional_string="_$(i-1)"
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i], parameters.measuredir, Dict[]; additional_string="_$(i-1)"
                )
            end
        end
    else
        gflow = GradientFlow(
            U,
            parameters.flow_integrator,
            parameters.flow_num,
            parameters.flow_steps,
            parameters.flow_tf;
            measure_every=parameters.flow_measure_every,
        )

        measurements = MeasurementMethods(U, parameters.measuredir, parameters.measurements)
        measurements_with_flow = MeasurementMethods(
            U, parameters.measuredir, parameters.measurements_with_flow; flow=true
        )
    end

    # initialize functor responsible for saving gaugefield configurations
    save_configs = SaveConfigs(
        parameters.saveU_format, parameters.saveU_dir, parameters.saveU_every
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
    fermion_action = univ.fermion_actions
    bias = univ.bias

    # load in config and recalculate gauge action if given
    load_gaugefield!(U, parameters) && (U.Sg = calc_gauge_action(U))

    @level1("┌ Thermalization:")
    _, runtime_therm = @timed begin
        for itrj in 1:(parameters.numtherm)
            @level1("|  itrj = $itrj")
            _, updatetime = @timed begin # time each update iteration
                update!(
                    updatemethod,
                    U;
                    fermion_action=fermion_action,
                    bias=nothing,
                    metro_test=false,
                )
            end
            @level1("|  Elapsed time:\t$(updatetime) [s]")
        end
    end

    @level1("└ Total elapsed time:\t$(runtime_therm) [s]\n")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    @level1("┌ Production:")
    _, runtime_all = @timed begin
        numaccepts = 0.0
        for itrj in 1:(parameters.numsteps)
            @level1("|  itrj = $itrj")

            _, updatetime = @timed begin
                accepted = update!(
                    updatemethod,
                    U;
                    fermion_action=fermion_action,
                    bias=bias,
                    metro_test=true,
                )
                rand() < 0.5 && update!(parity, U)
                accepted == true && update_bias!(bias, U.CV, itrj, true)
                numaccepts += accepted
            end

            print_acceptance_rates(numaccepts, itrj)
            @level1("|  Elapsed time:\t$(updatetime) [s]\n")

            save_gaugefield(save_configs, U, itrj)

            _, mtime = @timed calc_measurements(measurements, U, itrj)
            _, fmtime = @timed calc_measurements_flowed(
                measurements_with_flow, gflow, U, itrj
            )
            calc_weights(bias, U.CV, itrj)
            @level1(
                "|  Meas. elapsed time:     $(mtime)  [s]\n" *
                    "|  FlowMeas. elapsed time: $(fmtime) [s]"
            )
        end
    end

    @level1("└\nTotal elapsed time:\t$(convert_seconds(runtime_all))\n@ $(current_time())")
    flush(stdout)
    # close all the I/O streams
    close(updatemethod)
    close(measurements)
    close(measurements_with_flow)
    bias ≢ nothing && close(bias)
    close(Output.__GlobalLogger[])
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
    swap_every = parameters.swap_every
    rank0_updates = parameters.non_metadynamics_updates
    measure_on_all = parameters.measure_on_all

    # if stream 1 uses hmc then we have to recalc the CV before tempering
    uses_hmc = updatemethod isa HMC

    @level1("┌ Thermalization:")
    _, runtime_therm = @timed begin
        for itrj in 1:(parameters.numtherm)
            @level1("|  itrj = $itrj")
            _, updatetime = @timed begin
                for i in reverse(1:numinstances)
                    # thermalize all streams with the updatemethod of stream 1
                    # shouldnt be a problem for HMC, since we force 0-friction
                    # for thermalization updates and reverse the order, so stream 1 is last
                    update!(updatemethod, U[i]; bias=nothing, metro_test=false, friction=0)
                end
            end
            @level1("|  Elapsed time:\t$(updatetime) [s]")
        end
    end

    @level1("└ Total elapsed time:\t$(runtime_therm) [s]\n")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    @level1("┌ Production:")
    _, runtime_all = @timed begin
        numaccepts = zeros(numinstances)
        numaccepts_temper = zeros(Int64, numinstances - 1)

        for itrj in 1:(parameters.numsteps)
            @level1("|  itrj = $itrj")
            _, updatetime = @timed begin
                tmp = 0.0
                for _ in 1:rank0_updates
                    tmp += update!(updatemethod, U[1]; bias=nothing)
                end
                numaccepts[1] += tmp / rank0_updates
                rand() < 0.5 && update!(parity, U[1])

                for i in 2:numinstances
                    accepted = update!(updatemethod_pt, U[i]; bias=bias[i])
                    accepted == true && update_bias!(bias[i], U[i].CV, itrj, true)
                    numaccepts[i] += accepted
                end
            end

            print_acceptance_rates(numaccepts, itrj)
            @level1("|  Elapsed time:\t$(updatetime) [s]")

            temper!(U, bias, numaccepts_temper, swap_every, itrj; recalc=!uses_hmc)

            save_gaugefield(save_configs, U[1], itrj)

            _, mtime = @timed calc_measurements(measurements, U, itrj)
            _, fmtime = @timed calc_measurements_flowed(
                measurements_with_flow, gflow, U, itrj, measure_on_all
            )
            calc_weights(bias, [U[i].CV for i in 1:numinstances], itrj)
            @level1(
                "|  Meas. elapsed time:     $(mtime)  [s]\n" *
                    "|  FlowMeas. elapsed time: $(fmtime) [s]"
            )
        end
    end

    @level1("└\nTotal elapsed time:\t$(convert_seconds(runtime_all))\n@ $(current_time())")
    flush(stdout)
    close(updatemethod)
    [close(m) for m in measurements]
    [close(mf) for mf in measurements_with_flow]
    [close(b) for b in bias]
    close(Output.__GlobalLogger[])
    return nothing
end
