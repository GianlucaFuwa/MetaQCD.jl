function build_bias(filenamein::String; backend="cpu")
    # When using MPI we make sure that only rank 0 prints to the console
    if mpi_amroot()
        ext = splitext(filenamein)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    # load parameters from toml file
    parameters = construct_params_from_toml(filenamein; backend=backend)
    multi_sim = (prod(parameters.numprocs_cart) == 1) && mpi_parallel()
    mpi_barrier()

    if mpi_amroot()
        oneinst = parameters.numinstances == 1
        @assert multi_sim ⊻ oneinst "MPI must be enabled if and only if numinstances > 1, numinstances was $(parameters.numinstances)"
        @assert mpi_size() == parameters.numinstances "numinstances has to be equal to the number of MPI ranks"
        @assert parameters.kind_of_bias ∉ ("none", "parametric") "bias has to be \"metad\" or \"opes\" in build, was $(parameters.kind_of_bias)"
        @assert parameters.is_static == false "Bias cannot be static in build"
    end

    # set random seed if provided, otherwise generate one
    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed * (mpi_myrank() + 1))
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    logger_io = mpi_amroot() ? parameters.log_dir * "/logs.txt" : devnull
    set_global_logger!(parameters.verboselevel, logger_io; tc=parameters.log_to_console)

    # print time and system info, because it looks cool I guess
    # btw, all these "@level1" calls are just for logging, level1 is always printed
    # and anything higher has to specified in the parameter file (default is level2)
    @level1("Start time: @ $(current_time())")
    # buf = IOBuffer()
    # InteractiveUtils.versioninfo(buf)
    # versioninfo = String(take!(buf))
    # @level1(versioninfo)
    @level1("[ Running MetaQCD.jl version $(PACKAGE_VERSION)\n")
    @level1("[ Random seed is: $seed\n")

    if parameters.load_checkpoint
        univ_args..., updatemethod, _, _ = load_checkpoint(parameters.load_checkpoint_path)
        univ = Univ(univ_args...)
    else
        univ = Univ(parameters; use_mpi=multi_sim)
        updatemethod = nothing
    end

    build_bias!(univ, parameters, updatemethod)
    return nothing
end

function build_bias!(univ, parameters, updatemethod=nothing)
    U = univ.U

    if updatemethod === nothing
        updatemethod = Updatemethod(parameters, U)
    end

    gflow = GradientFlow(
        U,
        parameters.flow_integrator,
        parameters.flow_num,
        parameters.flow_steps,
        parameters.flow_tf;
        measure_every=parameters.flow_measure_every,
    )

    additional_string = "_$(mpi_myrank()+1)"

    measurements = MeasurementMethods(
        U,
        parameters.measure_dir,
        parameters.measurements;
        additional_string=additional_string,
    )

    measurements_with_flow = MeasurementMethods(
        U,
        parameters.measure_dir,
        parameters.measurements_with_flow;
        additional_string=additional_string,
        flow=true,
    )

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
)
    U = univ.U
    fermion_action = univ.fermion_action
    bias = univ.bias
    comm = mpi_comm()
    # This used to be in Bias itself, but I took all the IOBuffers away from structs
    # that are needed for checkpointing
    mpi_barrier()

    @level1("┌ Thermalization:")
    _, runtime_therm = @timed begin
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
            # @level1("|  Elapsed time:\t$(updatetime) [s] @ $(current_time())\n-")
        end
    end

    @level1("└ Total elapsed time:\t$(runtime_therm) [s]\n")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    mpi_barrier()

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
                numaccepts += accepted
            end

            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(current_time())\n")
            # all procs send their CVs to all other procs and update their copy of the bias
            CVs = mpi_allgather(U.CV::Float64, comm)
            accepteds = mpi_allgather(accepted::Bool, comm)
            accepted_CVs = CVs[findall(accepteds)] # update only on those CVs that were accepted
            update_bias!(bias, accepted_CVs, itrj)
            acceptances = mpi_allgather(numaccepts::Float64, comm) # XXX: should use MPI.gather?
            print_acceptance_rates(acceptances, itrj)

            create_checkpoint(checkpointer, univ, updatemethod, nothing, itrj)

            calc_measurements(measurements, U, itrj)
            calc_measurements_flowed(measurements_with_flow, gflow, U, itrj)
            calc_weights(bias, U.CV, itrj)
        end
    end

    # @level1("└\nTotal elapsed time:\t$(convert_seconds(runtime_all))\n@ $(current_time())")
    flush(stdout)
    # close all the I/O streams
    close(updatemethod)
    close(measurements)
    close(measurements_with_flow)
    close(bias)
    close(MetaIO.__GlobalLogger[])
    isinteractive() && set_global_logger!(1) # Reset logger if run from REPL
    return nothing
end
