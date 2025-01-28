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
    multi_sim = (prod(parameters.numprocs_cart) == 1) && mpi_parallel()
    mpi_barrier()

    if mpi_amroot()
        oneinst = parameters.numinstances == 1
        @assert mpi_size() == parameters.numinstances """
        numinstances has to be equal to the number of MPI ranks
        """
        @assert multi_sim ⊻ oneinst """
        MPI must be enabled only if numinstances > 1 or fields are distributed
        numinstances was: $(parameters.numinstances) but comm size was $(mpi_size())
        """
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
        univ_args..., updatemethod, _, _ = load_checkpoint(parameters.load_checkpoint_path)
        univ = Univ(univ_args...)
    else
        univ = Univ(parameters; mpi_multi_sim=multi_sim)
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

    additional_string = "_$(lpad(mpi_myrank(), 3, "0")).txt"

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
    comm = mpi_comm()
    starting_Q = parameters.starting_Q
    num_cv = length(bias)
    therm_cv = Matrix{Float64}(undef, num_cv, parameters.numtherm)
    adaptive_σ = is_adaptive(bias)

    @level1("- Thermalization:")
    _, runtime_therm = @timed begin
        !isnothing(starting_Q) && set_instanton!(U, starting_Q[mpi_myrank()+1])

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
            std_cv = mpi_allgather(std(view(therm_cv, i, :))::Float64, comm)
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

            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
            # all procs send their CVs to all other procs and update their copy of the bias
            CVs = mpi_allgather(tuple(U.CV...)::NTuple{num_cv,Float64}, comm)
            accepteds = mpi_allgather(accepted::Bool, comm)
            accepted_CVs = CVs[findall(accepteds)] # update only on those CVs that were accepted

            if !is_distributed(U) || mpi_amroot()
                update_bias!(bias, accepted_CVs, itrj; mpi_multi_sim=mpi_multi_sim)
            end

            acceptances = mpi_allgather(numaccepts::Float64, comm) # XXX: should use MPI.gather?
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
