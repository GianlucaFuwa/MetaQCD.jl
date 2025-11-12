function run_build(parameterfile::String)
    # When using MPI we make sure that only rank 0 prints to the console
    if mpi_amroot()
        ext = splitext(parameterfile)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    return run_build(construct_params_from_toml(parameterfile))
end

function run_build(parameters)
    if parameters.backend == "cuda"
        @assert "cuda" in keys(BACKENDS) """
        In order to use the CUDA Backend, CUDA.jl has to be loaded
        """
    elseif parameters.backend ∈ ("rocm", "roc", "amdgpu")
        @assert "rocm" in keys(BACKENDS) """
        In order to use the ROCM Backend, AMDGPU.jl has to be loaded
        """
    end

    @assert !parameters.tempering_enabled "Tempering must not be enabled in build"
    num_instances = parameters.numinstances
    starting_Q = parameters.starting_Q
    @assert isnothing(starting_Q) || (length(starting_Q) >= num_instances)
    num_dist = prod(parameters.numprocs_cart)

    multi_sim = if num_instances != 1
        @assert mpi_parallel() """
        Multiple walkers (numinstances>1) only possible with MPI enabled, i.e., number of ranks>=numinstances
        """
        true
    else
        false
    end

    @assert mpi_size() == num_instances * num_dist "MPI comm size must be = numinstances*prod(numprocs_cart)"
    color = instance_from_rank(mpi_myrank(), num_instances)
    mpi_split(mpi_comm(); color)
    MPI_NUMINSTANCES[] = num_instances # change global consant defined in utils/mpi.jl

    if mpi_amroot()
        oneinst = parameters.numinstances == 1
        @assert mpi_size() == parameters.numinstances*num_dist """
        numinstances*prod(numprocs_cart) has to be equal to the number of MPI ranks
        """
        if num_dist == 1
            @assert multi_sim ⊻ oneinst """
            MPI must be enabled only if numinstances > 1 or fields are distributed
            numinstances was: $(parameters.numinstances) but comm size was $(mpi_size())
            """
        end
        @assert length(parameters.biases) > 0 """
        There has to be at least one bias when in build mode
        """
    end

    # set random seed if provided, otherwise generate one
    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed * (mpi_myrank() + 1) % UInt64)
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

    @level1("# Working directory: $(pwd()) @ $(string(current_time()))")
    @level1("[ Running MetaQCD.jl version $(PACKAGE_VERSION)\n")
    @level1("[ Random seed is: $seed\n")

    if parameters.load_checkpoint_fromfile
        univ_args..., updatemethod, _, _ = load_checkpoint(parameters.load_checkpoint_path)
        univ = Univ(univ_args...; mpi_multi_sim=multi_sim, build=true)
    else
        univ = Univ(parameters; mpi_multi_sim=multi_sim, build=true)
        updatemethod = nothing
    end

    build_bias!(univ, parameters, updatemethod; mpi_multi_sim=multi_sim)
    return nothing
end

function build_bias!(univ, parameters, updatemethod; mpi_multi_sim=false)
    U = univ.U

    if isnothing(updatemethod)
        updatemethod = Updatemethod(parameters, U)
    end

    gflow = construct_flow(U, parameters)

    additional_string = "_$(lpad(MPI_INSTANCE[], 3, "0")).txt"

    measurements = MeasurementMethods(
        U,
        parameters.measure_dir,
        parameters.measurements;
        additional_string=additional_string,
    )

    measurements_with_flow = ntuple(length(gflow)) do i
        MeasurementMethods(
            U,
            parameters.measure_dir,
            parameters.measurements_with_flow;
            additional_string=additional_string,
            flow=gflow[i],
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

    mpi_barrier()
    metabuild!(
        parameters,
        univ,
        updatemethod,
        gflow,
        measurements,
        measurements_with_flow,
        config_saver,
        checkpointer,
        mpi_multi_sim,
    )
    return nothing
end

function metabuild!(
    parameters,
    univ,
    updatemethod,
    gflow,
    measurements,
    measurements_with_flow,
    config_saver,
    checkpointer,
    mpi_multi_sim,
)
    U = univ.U
    fermion_action = univ.fermion_action
    bias = univ.bias
    comm_shared = mpi_comm_shared()
    starting_Q = parameters.starting_Q
    num_cv = length(bias)
    therm_cv = Matrix{Float64}(undef, num_cv, parameters.numtherm)
    adaptive_σ = is_adaptive(bias)
    myinstance = MPI_INSTANCE[]
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

    load_field!(U, parameters)

    last_updatetime = 0.0 # look at last update time to determine whether we are going past the time limit

    @level1("- Thermalization:")
    _, runtime_therm = @timed begin
        !isnothing(starting_Q) && set_instanton!(U, starting_Q[myinstance+1])

        for itrj in 1:(parameters.numtherm)
            if (last_updatetime + time() + TIME_BUFFER - LOAD_TIME) > JOB_TIME_LIMIT
                break
            end

            @level1("|  itrj = $itrj")
            _, updatetime = @timed begin
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
                fp = fopen(logtimepath, "a")
                printf(fp, "%-.10E", updatetime)
                newline(fp)
                fclose(fp)
            end

            if any(adaptive_σ)
                recalc_cv!(U, bias)

                for icv in 1:num_cv
                    therm_cv[icv, itrj] = bias.CV[icv]
                end
            end

            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
        end
    end

    @level1("- Thermalization elapsed time:\t$(runtime_therm) [s]\n")
    recalc_cv!(U, bias) # need to recalc cv since it was not updated during therm

    mpi_barrier()

    for i in 1:num_cv
        if adaptive_σ[i]
            std_cv = mpi_allgather(std(view(therm_cv, i, :))::Float64, comm_shared)
            set_sigma0!(bias, mean(std_cv), i)
        end
    end

    @level1("- Production:")
    _, runtime_prod = @timed begin
        numaccepts = 0.0
        for itrj in 1:(parameters.numsteps)
            if (last_updatetime + time() + TIME_BUFFER - LOAD_TIME) > JOB_TIME_LIMIT
                break
            end

            @level1("|  itrj = $itrj")

            _, updatetime = @timed begin
                accepted = update!(
                    updatemethod,
                    U;
                    fermion_action=fermion_action,
                    bias=bias,
                    metro_test=true,
                )
                numaccepts += accepted
                mpi_barrier()
            end

            last_updatetime = updatetime

            if mpi_amroot(mpi_comm_instance())
                fp = fopen(logtimepath, "a")
                printf(fp, "%-.10E", updatetime)
                newline(fp)
                fclose(fp)
            end

            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
            # all procs send their CVs to all other procs and update their copy of the bias
            if parameters.recycle && updatemethod isa HMC
                sCVs = updatemethod.substep_CVs
                all_CVs = [bias.CV, view(sCVs, 2:length(sCVs))...]
                CVs = mpi_allgather(all_CVs, comm_shared)
                update_bias!(bias, CVs, itrj; mpi_multi_sim=mpi_multi_sim)
            else
                CVs = mpi_allgather(tuple(bias.CV...)::NTuple{num_cv,Float64}, comm_shared)
                update_bias!(bias, CVs, itrj; mpi_multi_sim=mpi_multi_sim)
            end

            acceptances = mpi_allgather(numaccepts::Float64, comm_shared) # XXX: should use MPI.gather?
            print_acceptance_rates(acceptances, itrj)

            save_field(config_saver, U, itrj, parameters)
            create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj)

            calc_measurements(measurements, U, itrj; mpi_multi_sim=mpi_multi_sim)
            calc_measurements_flowed(
                measurements_with_flow, gflow, U, itrj; mpi_multi_sim=mpi_multi_sim
            )
            calc_weights(bias, itrj)
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
