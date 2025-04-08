function run_sim(parameterfile::String; backend="cpu")
    # When using MPI we make sure that only rank 0 prints to the console
    if mpi_amroot()
        ext = splitext(parameterfile)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    # load parameters from toml file
    parameters = construct_params_from_toml(parameterfile; backend=backend)
    num_instances = parameters.numinstances
    num_dist = prod(parameters.numprocs_cart)

    multi_sim = if mpi_size() > num_dist
        @assert mpi_size() == num_instances * num_dist "MPI comm size must be = numinstances*prod(numprocs_cart)"
        true
    else
        @assert mpi_size() == num_dist """
        MPI comm size must be = prod(numprocs_cart) when not using multiple simulation streams or = numinstances*prod(numprocs_cart) when doing so 
        """
        false
    end

    if parameters.tempering_enabled
        @assert length(parameters.biases) > 0 """
        There has to be at least one bias when using tempering
        """
    end

    mpi_split(mpi_comm(); color=mpi_myrank()%num_instances)
    MPI_NUMINSTANCES[] = num_instances

    # set random seed if provided, otherwise generate one
    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!((seed * (mpi_myrank() + 1)) % UInt64)
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    logpath = if mpi_amroot(mpi_comm_instance())
        joinpath(parameters.log_dir, "logs_$(lpad(MPI_INSTANCE[], 3, "0")).txt")
    else
        nothing
    end

    to_console = mpi_amroot() ? parameters.log_to_console : false

    set_global_logger!(parameters.verboselevel, logpath; tc=to_console)

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
        univ = Univ(parameters; mpi_multi_sim=multi_sim)
        updatemethod = updatemethod_pt = nothing
    end

    run_sim!(univ, parameters, updatemethod, updatemethod_pt; mpi_multi_sim=multi_sim)
    return nothing
end

function run_sim!(univ, parameters, updatemethod, updatemethod_pt; mpi_multi_sim=false)
    U = univ.U

    # initialize update method, measurements, and bias
    if parameters.tempering_enabled
        if mpi_multi_sim
            if isnothing(updatemethod) && MPI_INSTANCE[]==0
                updatemethod = Updatemethod(parameters, U)
            elseif !isnothing(updatemethod) && MPI_INSTANCE[]==0
                # TODO:
            elseif isnothing(updatemethod_pt) && !(MPI_INSTANCE[]==0)
                faction_type = if univ.fermion_action == QuenchedFermionAction() 
                    QuenchedFermionAction
                else
                    typeof(univ.fermion_action[1])
                end
                # all MetaD streams use HMC, so there is no need to initialize more than 1
                hmc_integrator = parameters.hmc_integrator
                hmc_rafriction = parameters.hmc_rafriction
                updatemethod = HMC(
                    U,
                    integrator_from_str(hmc_integrator, hmc_rafriction),
                    parameters.hmc_trajectory,
                    parameters.hmc_steps,
                    parameters.hmc_friction,
                    parameters.hmc_numsmear_gauge,
                    parameters.hmc_numsmear_fermion,
                    parameters.hmc_rhostout_gauge,
                    parameters.hmc_rhostout_fermion;
                    hmc_logging=true,
                    fermion_action=faction_type,
                    heavy_flavours=length(parameters.Nf) - 1,
                    num_cv=length(univ.bias),
                    logdir=parameters.log_dir,
                    instance=MPI_INSTANCE[],
                )
            elseif !isnothing(updatemethod_pt) && !(MPI_INSTANCE[]==0)
                updatemethod = updatemethod_pt[MPI_INSTANCE[]]
            end

            parity = if parameters.parity_update && MPI_INSTANCE[]==0
                ParityUpdate(U)
            else
                nothing
            end
        else
            if isnothing(updatemethod) && isnothing(updatemethod_pt)
                updatemethod = Updatemethod(parameters, U[1])
                faction_type = if univ.fermion_action == QuenchedFermionAction() 
                    QuenchedFermionAction
                else
                    typeof(univ.fermion_action[1])
                end
                # all MetaD streams use HMC, so there is no need to initialize more than 1
                hmc_integrator = parameters.hmc_integrator
                hmc_rafriction = parameters.hmc_rafriction
                updatemethod_pt = HMC(
                    U[1],
                    integrator_from_str(hmc_integrator, hmc_rafriction),
                    parameters.hmc_trajectory,
                    parameters.hmc_steps,
                    parameters.hmc_friction,
                    parameters.hmc_numsmear_gauge,
                    parameters.hmc_numsmear_fermion,
                    parameters.hmc_rhostout_gauge,
                    parameters.hmc_rhostout_fermion;
                    hmc_logging=true,
                    fermion_action=faction_type,
                    heavy_flavours=length(parameters.Nf) - 1,
                    num_cv=length(univ.bias[1]),
                    logdir=parameters.log_dir,
                    instance=1:parameters.numinstances-1,
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

    numinstances = parameters.numinstances

    if parameters.tempering_enabled && !mpi_multi_sim
        gflow = construct_flow(U[1], parameters)
        measurements = Vector{MeasurementMethods}(undef, numinstances)
        measurements[1] = MeasurementMethods(
            U[1],
            parameters.measure_dir,
            parameters.measurements;
            additional_string="_000.txt",
        )

        for i in 2:numinstances
            if parameters.measure_on_all
                measurements[i] = MeasurementMethods(
                    U[i],
                    parameters.measure_dir,
                    parameters.measurements;
                    additional_string = "_$(lpad(i-1, 3, "0")).txt"
                )
            else
                measurements[i] = MeasurementMethods(U[i], parameters.measure_dir, Dict[])
            end
        end

        measurements_with_flow = ntuple(length(gflow)) do i
            _measurements_with_flow = Vector{MeasurementMethods}(undef, numinstances)
            _measurements_with_flow[1] = MeasurementMethods(
                U[1],
                parameters.measure_dir,
                parameters.measurements_with_flow;
                flow=gflow[i],
                additional_string="_000.txt",
            )

            for j in 2:numinstances
                if parameters.measure_on_all
                    _measurements_with_flow[j] = MeasurementMethods(
                        U[j],
                        parameters.measure_dir,
                        parameters.measurements_with_flow;
                        flow=gflow[i],
                        additional_string = "_$(lpad(j-1, 3, "0")).txt"
                    )
                else
                    _measurements_with_flow[j] = MeasurementMethods(
                        U[j],
                        parameters.measure_dir,
                        Dict[];
                        flow=gflow[i],
                    )
                end
            end

            _measurements_with_flow
        end
    else
        gflow = construct_flow(U, parameters)
        measurements = MeasurementMethods(
            U, parameters.measure_dir, parameters.measurements;
            additional_string="_$(lpad(MPI_INSTANCE[], 3, "0")).txt"
        )
        measurements_with_flow = ntuple(length(gflow)) do i
            MeasurementMethods(
                U, parameters.measure_dir, parameters.measurements_with_flow;
                flow=gflow[i], additional_string="_$(lpad(MPI_INSTANCE[], 3, "0")).txt"
            )
        end
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
            mpi_multi_sim,
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
    mpi_multi_sim,
)
    U = univ.U
    fermion_action = univ.fermion_action
    bias = univ.bias
    myinstance = MPI_INSTANCE
    tempering_enabled = parameters.tempering_enabled
    numaccepts_temper = zeros(Int64, univ.numinstances-1)
    instance_state = collect(0:univ.numinstances)
    swap_every = parameters.swap_every
    # INFO: Log times per update in seconds
    logtimepath = if mpi_amroot(mpi_comm_instance())
        joinpath(parameters.log_dir, "timings_$(lpad(MPI_INSTANCE[], 3, "0")).txt")
    else
        nothing
    end

    if !isnothing(logtimepath)
        fp = fopen(logtimepath, "w")
        printf(fp, "%s", "time [s]")
        newline(fp)
        fclose(fp)
    end

    # load in config and recalculate gauge action if given
    load_config!(U, parameters) && (U.Sg = calc_gauge_action(U))

    @level1("- Thermalization:")
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

            mpi_barrier()

            if mpi_amroot(mpi_comm_instance())
                fp = fopen(logtimepath, "a")
                printf(fp, "%-.10E", updatetime)
                newline(fp)
                fclose(fp)
            end

            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))\n-")
        end
    end

    @level1("-- Thermalization elapsed time:\t$(runtime_therm) [s]\n")
    recalc_cv!(U, bias) # need to recalc cv since it was not updated during therm

    mpi_barrier()

    @level1("- Production:")
    _, runtime_prod = @timed begin
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

                if accepted
                    update_bias!(
                        bias, U.CV, itrj, myinstance[]; mpi_multi_sim=mpi_multi_sim
                    )
                end

                numaccepts += accepted
            end

            mpi_barrier()

            if mpi_amroot(mpi_comm_instance())
                fp = fopen(logtimepath, "a")
                printf(fp, "%-.10E", updatetime)
                newline(fp)
                fclose(fp)
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

            _, mtime = @timed calc_measurements(
                measurements, U, itrj; mpi_multi_sim=mpi_multi_sim
            )
            _, fmtime = @timed for i in eachindex(gflow)
                calc_measurements_flowed(
                    measurements_with_flow[i], gflow[i], U, itrj;
                    mpi_multi_sim=mpi_multi_sim
                )
            end
            calc_weights(bias, U.CV, itrj; mpi_multi_sim=mpi_multi_sim)
            @level1("|  Meas. elapsed time:     $(mtime)  [s]")
            @level1("|  FlowMeas. elapsed time: $(fmtime) [s]\n-")
        end
    end

    @level1("- Production elapsed time:\t$(runtime_prod) [s]\n")
    print_total_time(runtime_therm + runtime_prod)
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

    @level1("- Thermalization:")
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
                        fermion_action=fermion_action,
                        bias=NoBias(),
                        metro_test=false,
                        therm=true,
                        instance=i-1,
                    )
                end
            end
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
        end
    end

    @level1("-- Thermalization elapsed time:\t$(runtime_therm) [s]\n")
    recalc_cv!(U, bias) # need to recalc cv since it was not updated during therm

    @level1("- Production:")
    _, runtime_prod = @timed begin
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
                        instance=0,
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
                        instance=i-1,
                    )
                    accepted == true && update_bias!(bias[i], U[i].CV, itrj)
                    numaccepts[i] += accepted
                end
            end

            print_acceptance_rates(numaccepts, itrj)
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")

            temper!(U, bias, numaccepts_temper, swap_every, itrj; recalc=true)

            save_config(config_saver, U[1], itrj, parameters)
            create_checkpoint(checkpointer, univ, updatemethod, updatemethod_pt, itrj)

            _, mtime = @timed calc_measurements(measurements, U, itrj, measure_on_all)
            _, fmtime = @timed for i in eachindex(gflow)
                calc_measurements_flowed(
                    measurements_with_flow[i], gflow[i], U, itrj, measure_on_all
                )
            end
            calc_weights(bias, [U[i].CV for i in 1:numinstances], itrj)
            @level1("|  Meas. elapsed time:     $(mtime)  [s]")
            @level1("|  FlowMeas. elapsed time: $(fmtime) [s]\n-")
        end
    end

    @level1("- Production elapsed time:\t$(runtime_prod) [s]\n")
    print_total_time(runtime_therm + runtime_prod)
    flush(stdout)
    close(MetaIO.__GlobalLogger[])
    isinteractive() && set_global_logger!(1) # Reset logger if run from REPL
    return nothing
end
