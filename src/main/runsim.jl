function run_sim(filenamein::String; backend="cpu")
    # When using MPI we make sure that only rank 0 prints to the console
    if mpi_amroot()
        ext = splitext(filenamein)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    # load parameters from toml file
    parameters = construct_params_from_toml(filenamein; backend=backend)
    mpi_barrier()

    if mpi_amroot()
        if parameters.tempering_enabled
            @assert parameters.kind_of_bias != "none" "bias cannot be \"none\" in parallel tempering"
        end
    end

    # set random seed if provided, otherwise generate one
    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!((seed * (mpi_myrank() + 1)) % UInt64)
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    logpath = mpi_amroot() ? joinpath(parameters.log_dir, "logs.txt") : nothing
    set_global_logger!(
        parameters.verboselevel, logpath; tc=parameters.log_to_console
    )

    # print time and system info, because it looks cool I guess
    # btw, all these "@level1" calls are just for logging, level1 is always printed
    # and anything higher has to specified in the parameter file (default is level2)
    @level1("# Working directory: $(pwd()) @ $(string(current_time()))")
    # buf = IOBuffer()
    # InteractiveUtils.versioninfo(buf)
    # versioninfo = String(take!(buf))
    # @level1(versioninfo)
    @level1("[ Running MetaQCD.jl version $(PACKAGE_VERSION)\n")
    @level1("[ Random seed is: $seed\n")

    if parameters.load_checkpoint_fromfile
        univ_args..., updatemethod, updatemethod_pt, _ = load_checkpoint(
            parameters.load_checkpoint_path
        )
        univ = Univ(univ_args...)
    else
        univ = Univ(parameters)
        updatemethod = updatemethod_pt = nothing
    end

    run_sim!(univ, parameters, updatemethod, updatemethod_pt)
    return nothing
end

function run_sim!(
    univ::Univ, parameters::ParameterSet, updatemethod, updatemethod_pt; mpi_multi_sim=false
)
    U = univ.U

    # initialize update method, measurements, and bias
    if parameters.tempering_enabled
        if mpi_multi_sim
            if isnothing(updatemethod)
                updatemethod = Updatemethod(parameters, U)
            end

            parity = parameters.parity_update ? ParityUpdate(U) : nothing
        else
            if isnothing(updatemethod) && isnothing(updatemethod_pt)
                updatemethod = Updatemethod(parameters, U[1])
                faction_type = if univ.fermion_action == QuenchedFermionAction() 
                    QuenchedFermionAction
                else
                    typeof(univ.fermion_action[1])
                end
                # all MetaD streams use HMC, so there is no need to initialize more than 1
                updatemethod_pt = HMC(
                    U[1],
                    integrator_from_str(parameters.hmc_integrator),
                    parameters.hmc_trajectory,
                    parameters.hmc_steps,
                    parameters.hmc_friction,
                    parameters.hmc_numsmear_gauge,
                    parameters.hmc_numsmear_fermion,
                    parameters.hmc_rhostout_gauge,
                    parameters.hmc_rhostout_fermion;
                    hmc_logging=false,
                    fermion_action=faction_type,
                    heavy_flavours=length(parameters.Nf) - 1,
                    bias_enabled=true,
                )
            end
            parity = parameters.parity_update ? ParityUpdate(U[1]) : nothing
        end
    else
        if isnothing(updatemethod)
            updatemethod = Updatemethod(parameters, U)
        end
        parity = parameters.parity_update ? ParityUpdate(U) : nothing
    end

    isnothing(parity) || @level1("[ Parity update enabled\n")

    if parameters.tempering_enabled && !mpi_multi_sim
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
            U[1], parameters.measure_dir, parameters.measurements; additional_string="_0"
        )

        measurements_with_flow[1] = MeasurementMethods(
            U[1],
            parameters.measure_dir,
            parameters.measurements_with_flow;
            flow=true,
            additional_string="_0",
        )
        for i in 2:(parameters.numinstances)
            if parameters.measure_on_all
                measurements[i] = MeasurementMethods(
                    U[i],
                    parameters.measure_dir,
                    parameters.measurements;
                    additional_string="_$(i-1)",
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i],
                    parameters.measure_dir,
                    parameters.measurements_with_flow;
                    flow=true,
                    additional_string="_$(i-1)",
                )
            else
                measurements[i] = MeasurementMethods(
                    U[i], parameters.measure_dir, Dict[]; additional_string="_$(i-1)"
                )
                measurements_with_flow[i] = MeasurementMethods(
                    U[i], parameters.measure_dir, Dict[]; additional_string="_$(i-1)"
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

        measurements = MeasurementMethods(U, parameters.measure_dir, parameters.measurements)
        measurements_with_flow = MeasurementMethods(
            U, parameters.measure_dir, parameters.measurements_with_flow; flow=true
        )
    end

    # initialize functor responsible for saving gaugefield configurations
    config_saver = ConfigSaver(
        parameters.save_config_format,
        parameters.save_config_dir,
        parameters.save_config_every,
    )

    checkpointer = Checkpointer(
        parameters.ensemble_dir, parameters.save_checkpoint_every
    )

    if parameters.tempering_enabled && !mpi_multi_sim
        metaqcd_PT!(
            parameters,
            univ,
            updatemethod,
            updatemethod_pt,
            gflow,
            measurements,
            measurements_with_flow,
            parity,
            config_saver,
            checkpointer,
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
            config_saver,
            checkpointer,
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
    config_saver,
    checkpointer,
)
    U = univ.U
    fermion_action = univ.fermion_action
    bias = univ.bias
    myinstance = univ.myinstance
    tempering_enabled = parameters.tempering_enabled
    numaccepts_temper = zeros(Int64, univ.numinstances-1)
    instance_state = collect(0:univ.numinstances-1)
    swap_every = parameters.swap_every

    # load in config and recalculate gauge action if given
    load_config!(U, parameters) && (U.Sg = calc_gauge_action(U))

    @level1("┌ Thermalization:")
    _, runtime_therm = @timed begin
        for itrj in 1:(parameters.numtherm)
            @level1("|  itrj = $itrj")
            _, updatetime = @timed begin # time each update iteration
                update!(
                    updatemethod,
                    U;
                    fermion_action=fermion_action,
                    bias=NoBias(),
                    metro_test=itrj>10, # So we dont get stuck at the beginning
                    therm=true,
                )
            end
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))\n-")
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
                accepted == true && update_bias!(bias, U.CV, itrj)
                numaccepts += accepted
            end

            print_acceptance_rates(numaccepts, itrj)
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")

            if tempering_enabled
                temper!(
                    U,
                    bias,
                    numaccepts_temper,
                    instance_state,
                    myinstance,
                    swap_every,
                    itrj;
                    recalc=(myinstance[]==0)
                )
            end

            save_config(config_saver, U, itrj, parameters)
            create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj)

            _, mtime = @timed calc_measurements(measurements, U, itrj, myinstance[])
            _, fmtime = @timed calc_measurements_flowed(
                measurements_with_flow, gflow, U, itrj, myinstance[]
            )
            calc_weights(bias, U.CV, itrj)
            @level1("|  Meas. elapsed time:     $(mtime)  [s]")
            @level1("|  FlowMeas. elapsed time: $(fmtime) [s]\n-")
        end
    end

    print_total_time(runtime_all)
    flush(stdout)
    close(MetaIO.__GlobalLogger[])
    isinteractive() && set_global_logger!(1) # Reset logger if run from REPL
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
    config_saver,
    checkpointer,
)
    numinstances = parameters.numinstances
    U = univ.U
    bias = univ.bias
    fermion_action = univ.fermion_action
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
                    update!(
                        updatemethod,
                        U[i];
                        fermion_action=fermion_action[i],
                        bias=NoBias(),
                        metro_test=false,
                        friction=0,
                    )
                end
            end
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
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
                    tmp += update!(
                        updatemethod,
                        U[1];
                        fermion_action=fermion_action,
                        bias=NoBias(),
                        metro_test=true,
                    )
                end
                numaccepts[1] += tmp / rank0_updates
                rand() < 0.5 && update!(parity, U[1])

                for i in 2:numinstances
                    accepted = update!(
                        updatemethod_pt,
                        U[i];
                        fermion_action=fermion_action,
                        bias=bias[i],
                        metro_test=true,
                    )
                    accepted == true && update_bias!(bias[i], U[i].CV, itrj)
                    numaccepts[i] += accepted
                end
            end

            print_acceptance_rates(numaccepts, itrj)
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")

            temper!(U, bias, numaccepts_temper, swap_every, itrj; recalc=!uses_hmc)

            save_config(config_saver, U[1], itrj, parameters)
            create_checkpoint(checkpointer, univ, updatemethod, updatemethod_pt, itrj)

            _, mtime = @timed calc_measurements(measurements, U, itrj)
            _, fmtime = @timed calc_measurements_flowed(
                measurements_with_flow, gflow, U, itrj, measure_on_all
            )
            calc_weights(bias, [U[i].CV for i in 1:numinstances], itrj)
            @level1("|  Meas. elapsed time:     $(mtime)  [s]")
            @level1("|  FlowMeas. elapsed time: $(fmtime) [s]\n-")
        end
    end

    print_total_time(runtime_all)
    flush(stdout)
    close(MetaIO.__GlobalLogger[])
    isinteractive() && set_global_logger!(1) # Reset logger if run from REPL
    return nothing
end
