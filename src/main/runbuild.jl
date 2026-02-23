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

    mpi_multi_sim = if num_instances != 1
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
            @assert mpi_multi_sim ⊻ oneinst """
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

    if parameters.load_checkpoint_path != ""
        rank = mpi_myrank(mpi_comm_instance())
        univ_args..., updatemethod, _, itrj = load_checkpoint(
            parameters; rank, mpi_multi_sim, build=true
        )
        univ = Univ(univ_args...)
    else
        itrj = nothing
        univ = Univ(parameters; mpi_multi_sim, build=true)
        updatemethod = nothing
    end

    @level1("[ Random seed is: $(string(copy(Random.default_rng())))\n")
    build_bias!(univ, parameters, updatemethod; mpi_multi_sim, itrj)
    return nothing
end

function build_bias!(univ, parameters, updatemethod; mpi_multi_sim=false, itrj=nothing)
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
        additional_string,
    )

    measurements_with_flow = ntuple(length(gflow)) do i
        MeasurementMethods(
            U,
            parameters.measure_dir,
            parameters.measurements_with_flow;
            additional_string,
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
        joinpath(parameters.ensemble_dir, "checkpoint"), parameters.save_checkpoint_every
    )

    timing_datafile = if mpi_amroot(mpi_comm_instance())
        SStaticString(
            joinpath(parameters.log_dir, "timings_$(lpad(MPI_INSTANCE[], 3, "0")).txt")
        )
    else
        nothing
    end

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
        timing_datafile,
        mpi_multi_sim,
        itrj,
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
    timing_datafile,
    mpi_multi_sim,
    starting_itrj=nothing
)
    U = univ.U
    fermion_action = univ.fermion_action
    bias = univ.bias
    comm_shared = mpi_comm_shared()
    starting_Q = parameters.starting_Q
    num_cv = length(bias)
    therm_cv = Matrix{Float64}(undef, num_cv, parameters.numtherm)
    adaptive_σ = is_adaptive(bias)
    rank = mpi_myrank(mpi_comm_instance())
    # len = length(updatemethod.substep_CVs)
    # CV_sendbuf = Vector{NTuple{num_cv,Float64}}(undef, len)
    # CV_recvbuf = Vector{NTuple{num_cv,Float64}}(undef, MPI_NUMINSTANCES[] * len)
    # global_accepts = Vector{Bool}(undef, MPI_NUMINSTANCES[])

    if !isnothing(timing_datafile)
        fp = fopen(timing_datafile, "w")
        printf(fp, printfmt(String), "time [s]")
        newline(fp)
        fclose(fp)
    end

    load_field!(U, parameters)

    last_updatetime = 0.0 # look at last update time to determine whether we are going past the time limit
    all_load_times = mpi_allgather(LOAD_TIME::Float64, mpi_comm())
    load_time = minimum(all_load_times)

    if isnothing(starting_itrj)
        @level2("- Thermalization:")
        _, runtime_therm = @timed begin
            set_instanton!(U, starting_Q)

            for itrj in 1:(parameters.numtherm)
                all_last_updatetime = mpi_allgather(last_updatetime, mpi_comm())
                if any(x -> x > JOB_TIME_LIMIT, all_last_updatetime .+ time() .+ TIME_BUFFER .- load_time)
                    @level1(
                        """### Run terminated before production trajectory $(itrj)
                        ### because time limit would be passed"""
                    )
                    create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj; rank)
                    mpi_barrier()
                    break
                end

                @level2("|  itrj = $itrj")
                _, updatetime = @timed begin
                    update!(
                        updatemethod,
                        U;
                        fermion_action,
                        bias=NoBias(),
                        metro_test=itrj>20, # So we dont get stuck at the beginning
                        therm=Val(true),
                    )
                    mpi_barrier()
                end

                last_updatetime = updatetime

                if mpi_amroot(mpi_comm_instance())
                    fp = fopen(timing_datafile, "a")
                    printf(fp, StaticString("%-.10E"), updatetime)
                    newline(fp)
                    fclose(fp)
                end

                if any(adaptive_σ)
                    recalc_cv!(U, bias)

                    for icv in 1:num_cv
                        therm_cv[icv, itrj] = bias.CV[icv]
                    end
                end

                @level2("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
            end
        end

        @level1("- Thermalization elapsed time:\t$(runtime_therm) [s]\n")
        recalc_cv!(U, bias) # need to recalc cv since it was not updated during therm
    else
        runtime_therm = 0.0
    end

    mpi_barrier()

    for i in 1:num_cv
        if adaptive_σ[i]
            std_cv = mpi_allgather(std(view(therm_cv, i, :))::Float64, comm_shared)
            set_sigma0!(bias, mean(std_cv), i)
        end
    end

    itrj_range = if isnothing(starting_itrj)
        1:parameters.numsteps
    else
        1+starting_itrj:(parameters.numsteps)+starting_itrj
    end

    @level2("- Production:")
    _, runtime_prod = @timed begin
        numaccepts = 0.0
        numitrj = 0

        for itrj in itrj_range
            all_last_updatetime = mpi_allgather(last_updatetime, mpi_comm())
            if any(x -> x > JOB_TIME_LIMIT, all_last_updatetime .+ time() .+ TIME_BUFFER .- load_time)
                @level1(
                    """### Run terminated before production trajectory $(itrj)
                    ### because time limit would be passed"""
                )
                create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj; rank)
                mpi_barrier()
                break
            end

            numitrj += 1
            @level2("|  itrj = $itrj")

            acc, updatetime = @timed begin
                accepted = update!(updatemethod, U; fermion_action, bias, metro_test=true)
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

            @level2("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")

            # all procs send their CVs to all other procs and update their copy of the bias
            substep_CVs = updatemethod.substep_CVs
            update_bias!(bias, substep_CVs, Bool(acc), itrj)

            print_acceptance_rates(numaccepts, numitrj)

            save_field(config_saver, U, itrj, parameters)
            create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj; rank)

            calc_measurements(measurements, U, itrj; mpi_multi_sim)
            calc_measurements_flowed(measurements_with_flow, gflow, U, itrj; mpi_multi_sim)
            calc_weights(bias, itrj)
        end
    end

    @level2("- Production elapsed time:\t$(runtime_prod) [s]\n")
    print_total_time(runtime_therm + runtime_prod)
    flush(stdout)
    close(MetaIO.__GlobalLogger[])
    isinteractive() && set_global_logger!(1) # Reset logger if run from REPL
    mpi_barrier()
    return nothing
end
