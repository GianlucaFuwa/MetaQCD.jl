function build_bias(parameterfile::String; backend="cpu")
    # When using MPI we make sure that only rank 0 prints to the console
    if mpi_amroot()
        ext = splitext(parameterfile)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    # load parameters from toml file
    parameters = construct_params_from_toml(parameterfile; backend=backend)

    @assert !parameters.tempering_enabled "Tempering must not be enabled in build"
    num_instances = parameters.numinstances
    starting_Q = parameters.starting_Q
    @assert isnothing(starting_Q) || (length(starting_Q) == num_instances)
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
    mpi_split(mpi_comm(); color=mpi_myrank()%num_instances)
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

    # print time and system info, because it looks cool I guess
    # all these "@level1" calls are just for logging, level1 is always printed
    # and anything higher has to specified in the parameter file (default is level2)
    @level1("# Working directory: $(pwd()) @ $(string(current_time()))")
    # buf = IOBuffer()
    # InteractiveUtils.versioninfo(buf)
    # versioninfo = String(take!(buf))
    # @level1(versioninfo)
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
    checkpointer,
    mpi_multi_sim,
)
    U = univ.U
    fermion_action = univ.fermion_action
    bias = univ.bias
    comm_root = mpi_comm_root()
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

    @level1("- Thermalization:")
    _, runtime_therm = @timed begin
        !isnothing(starting_Q) && set_instanton!(U, starting_Q[myinstance+1])

        for itrj in 1:(parameters.numtherm)
            @level1("|  itrj = $itrj")
            _, updatetime = @timed begin
                update!(
                    updatemethod,
                    U;
                    fermion_action=fermion_action,
                    bias=NoBias(),
                    metro_test=itrj>10, # So we dont get stuck at the beginning
                    therm=true,
                )

                mpi_barrier()
            end

            if mpi_amroot(mpi_comm_instance())
                fp = fopen(logtimepath, "a")
                printf(fp, "%-.10E", updatetime)
                newline(fp)
                fclose(fp)
            end

            if any(adaptive_σ)
                recalc_cv!(U, bias)

                for icv in 1:num_cv
                    therm_cv[icv, itrj] = U.CV[icv]
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
            std_cv = mpi_allgather(std(view(therm_cv, i, :))::Float64, comm_root)
            set_sigma0!(bias, mean(std_cv), i)
        end
    end

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
                numaccepts += accepted
            end

            mpi_barrier()

            if mpi_amroot(mpi_comm_instance())
                fp = fopen(logtimepath, "a")
                printf(fp, "%-.10E", updatetime)
                newline(fp)
                fclose(fp)
            end

            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
            # all procs send their CVs to all other procs and update their copy of the bias
            CVs = mpi_allgather(tuple(U.CV...)::NTuple{num_cv,Float64}, comm_root)
            accepteds = mpi_allgather(accepted::Bool, comm_root)
            accepted_CVs = CVs[findall(accepteds)] # update only on those CVs that were accepted

            update_bias!(bias, accepted_CVs, itrj; mpi_multi_sim=mpi_multi_sim)

            acceptances = mpi_allgather(numaccepts::Float64, comm_root) # XXX: should use MPI.gather?
            print_acceptance_rates(acceptances, itrj)

            create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj)

            calc_measurements(measurements, U, itrj; mpi_multi_sim=mpi_multi_sim)
            calc_measurements_flowed(
                measurements_with_flow, gflow, U, itrj; mpi_multi_sim=mpi_multi_sim
            )
            calc_weights(bias, U.CV, itrj)
        end
    end

    @level1("- Production elapsed time:\t$(runtime_prod) [s]\n")
    print_total_time(runtime_therm + runtime_prod)
    flush(stdout)
    close(MetaIO.__GlobalLogger[])
    isinteractive() && set_global_logger!(1) # Reset logger if run from REPL
    return nothing
end
