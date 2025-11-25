function run_sim(parameterfile::String)
    # When using MPI we make sure that only rank 0 prints to the console
    if mpi_amroot()
        ext = splitext(parameterfile)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    return run_sim(construct_params_from_toml(parameterfile))
end

function run_sim(parameters)
    if parameters.backend == "cuda"
        @assert "cuda" in keys(BACKENDS) """
        In order to use the CUDA Backend, CUDA.jl has to be loaded
        """
    elseif parameters.backend ∈ ("rocm", "roc", "amdgpu")
        @assert "rocm" in keys(BACKENDS) """
        In order to use the ROCM Backend, AMDGPU.jl has to be loaded
        """
    end

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

    color = multi_sim ? instance_from_rank(mpi_myrank(), num_instances) : 0
    mpi_split(mpi_comm(); color)
    MPI_NUMINSTANCES[] = num_instances # change global consant defined in utils/mpi.jl

    # set random seed if provided, otherwise generate one
    if parameters.randomseed != 0
        seed = parameters.randomseed
        if seed isa Vector
            my_local_rank = mpi_myrank(mpi_comm_instance())
            Random.seed!(((seed[MPI_INSTANCE[]+1]) * my_local_rank) % UInt64)
        else
            Random.seed!((seed * (mpi_myrank() + 1)) % UInt64)
        end
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

    if parameters.load_checkpoint_fromfile
        rank = mpi_myrank(mpi_comm_instance())
        univ_args..., updatemethod, _, itrj = load_checkpoint(parameters; rank)
        univ = Univ(univ_args...)
    else
        itrj = nothing
        univ = Univ(parameters; mpi_multi_sim=multi_sim)
        updatemethod = updatemethod_pt = nothing
    end

    @level1("[ Random seed is: $(string(copy(Random.default_rng())))\n")
    run_sim!(univ, parameters, updatemethod, updatemethod_pt, multi_sim)
    return nothing
end

function run_sim!(
    univ::Univ, 
    parameters::ParameterSet,
    updatemethod,
    updatemethod_pt,
    mpi_multi_sim=false,
    itrj=nothing,
)
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
                    "quenched"
                else
                    parameters.fermion_action
                end
                # all MetaD streams use HMC, so there is no need to initialize more than 1
                updatemethod = HMC(
                    U,
                    parameters.levels,
                    parameters.hmc_trajectory,
                    parameters.hmc_friction,
                    parameters.hmc_numsmear_gauge,
                    parameters.hmc_numsmear_fermion,
                    parameters.hmc_rhostout_gauge,
                    parameters.hmc_rhostout_fermion;
                    rafriction=parameters.hmc_rafriction,
                    hmc_logging=true,
                    fermion_action=faction_type,
                    numfermions=length(parameters.fermions),
                    numcv=length(parameters.biases),
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
                    "quenched"
                else
                    parameters.fermion_action
                end
                # all MetaD streams use HMC, so there is no need to initialize more than 1
                updatemethod_pt = HMC(
                    U[1],
                    parameters.levels,
                    parameters.hmc_trajectory,
                    parameters.hmc_friction,
                    parameters.hmc_numsmear_gauge,
                    parameters.hmc_numsmear_fermion,
                    parameters.hmc_rhostout_gauge,
                    parameters.hmc_rhostout_fermion;
                    rafriction=parameters.hmc_rafriction,
                    hmc_logging=true,
                    fermion_action=faction_type,
                    numfermions=length(parameters.fermions),
                    numcv=length(parameters.biases),
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
        joinpath(parameters.ensemble_dir, "checkpoint"), parameters.save_checkpoint_every
    )

    # INFO: Log times per update in seconds
    timing_datafile = if mpi_amroot(mpi_comm_instance())
        SStaticString(
            joinpath(parameters.log_dir, "timings_$(lpad(MPI_INSTANCE[], 3, "0")).txt")
        )
    else
        nothing
    end

    mpi_barrier()

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
            timing_datafile,
            mpi_multi_sim,
            Val(parameters.tempering_enabled),
            itrj,
        )
    end

    return nothing
end

function metaqcd!(
    parameters::ParameterSet,
    univ::Univ,
    updatemethod,
    gflow,
    measurements::MeasurementMethods,
    measurements_with_flow,
    parity,
    config_saver::ConfigSaver,
    checkpointer::Checkpointer,
    timing_datafile,
    mpi_multi_sim::Bool,
    ::Val{tempering_enabled},
    starting_itrj=nothing,
) where {tempering_enabled}
    U = univ.U
    fermion_action = univ.fermion_action
    bias = univ.bias
    numaccepts_temper = zeros(Int64, MPI_NUMINSTANCES[]-1)
    instance_state = collect(0:univ.numinstances)
    swap_every = parameters.swap_every

    if !isnothing(timing_datafile)
        fp = fopen(timing_datafile, "w")
        printf(fp, printfmt(String), "time [s]")
        newline(fp)
        fclose(fp)
    end

    # load in config and recalculate gauge action if given
    load_field!(U, parameters)

    last_updatetime = 0.0

    if isnothing(starting_itrj)
        @level1("- Thermalization:")
        _, runtime_therm = @timed begin
            for itrj in 1:(parameters.numtherm)
                if (last_updatetime + time() + TIME_BUFFER - LOAD_TIME) > JOB_TIME_LIMIT
                    break
                end

                @level1("|  itrj = $itrj")
                _, updatetime = @timed begin # time each update iteration
                    update!(
                        updatemethod,
                        U;
                        fermion_action=fermion_action,
                        bias=NoBias(),
                        metro_test=itrj>20, # So we dont get stuck at the beginning
                        therm=Val(true),
                    )
                    mpi_barrier()
                end

                last_updatetime = updatetime

                if mpi_amroot(mpi_comm_instance())
                    if !isnothing(timing_datafile)
                        fp = fopen(timing_datafile, "a")
                        printf(fp, StaticString("%-.10E"), updatetime)
                        newline(fp)
                        fclose(fp)
                    end
                end

                @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))\n-")
            end
        end

        @level1("-- Thermalization elapsed time:\t$(runtime_therm) [s]\n")
        recalc_cv!(U, bias) # need to recalc cv since it was not updated during therm
    else
        runtime_therm = 0.0
    end

    mpi_barrier()

    itrj_range = if isnothing(starting_itrj)
        1:parameters.numsteps
    else
        1+starting_itrj:(parameters.numsteps)+starting_itrj
    end

    @level1("- Production:")
    _, runtime_prod = @timed begin
        numaccepts = 0.0
        numitrj = 0
        for itrj in itrj_range
            if (last_updatetime + time() + TIME_BUFFER - LOAD_TIME) > JOB_TIME_LIMIT
                break
            end

            numitrj += 1
            @level1("|  itrj = $itrj")

            _, updatetime = @timed begin
                accepted = update!(
                    updatemethod,
                    U;
                    fermion_action=fermion_action,
                    bias=bias,
                    metro_test=true,
                )

                if rand() < 0.5
                    update!(parity, U)
                end

                update_bias!(bias, itrj)
                numaccepts += accepted
                mpi_barrier()
                accepted
            end

            last_updatetime = updatetime

            if mpi_amroot(mpi_comm_instance())
                if !isnothing(timing_datafile)
                    set_ext!(timing_datafile)
                    fp = fopen(timing_datafile, "a")
                    printf(fp, StaticString("%-.10E"), updatetime)
                    newline(fp)
                    fclose(fp)
                end
            end

            print_acceptance_rates(numaccepts, itrj)
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")

            if tempering_enabled
                temper!(
                    U,
                    bias,
                    numaccepts_temper,
                    instance_state,
                    swap_every,
                    itrj;
                    recalc=(MPI_INSTANCE[]==0)
                )
            end

            save_field(config_saver, U, itrj, parameters)
            create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj; rank)

            _, mtime = @timed calc_measurements(
                measurements, U, itrj; mpi_multi_sim=mpi_multi_sim
            )
            _, fmtime = @timed for i in eachindex(gflow)
                calc_measurements_flowed(
                    measurements_with_flow[i], gflow[i], U, itrj;
                    mpi_multi_sim=mpi_multi_sim
                )
            end

            calc_weights(bias, itrj; mpi_multi_sim=mpi_multi_sim)
            @level1("|  Meas. elapsed time:     $(mtime)  [s]")
            @level1("|  FlowMeas. elapsed time: $(fmtime) [s]\n-")
        end
    end

    @level1("- Production elapsed time:\t$(runtime_prod) [s]\n")
    print_total_time(runtime_therm + runtime_prod)
    flush(stdout)
    close(MetaIO.__GlobalLogger[])
    isinteractive() && set_global_logger!(1) # Reset logger if run from REPL
    mpi_barrier()
    return nothing
end

function metaqcd_PT!(
    parameters::ParameterSet,
    univ::Univ,
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
                        therm=Val(true),
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

                if rand() < 0.5
                    update!(parity, U[1])
                end

                for i in 2:numinstances
                    accepted = update!(
                        updatemethod_pt,
                        U[i];
                        fermion_action=fermion_action,
                        bias=bias[i],
                        metro_test=true,
                        instance=i-1,
                    )
                    update_bias!(bias[i], itrj)
                    numaccepts[i] += accepted
                end
            end

            for numaccepts_i in numaccepts
                print_acceptance_rates(numaccepts_i, itrj)
            end

            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")

            temper!(U, bias, numaccepts_temper, swap_every, itrj; recalc=true)

            save_field(config_saver, U[1], itrj, parameters)
            create_checkpoint(checkpointer, univ, updatemethod, updatemethod_pt, itrj; rank)

            _, mtime = @timed calc_measurements(measurements, U, itrj, measure_on_all)
            _, fmtime = @timed for i in eachindex(gflow)
                calc_measurements_flowed(
                    measurements_with_flow[i], gflow[i], U, itrj, measure_on_all
                )
            end
            calc_weights(bias, itrj)
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
