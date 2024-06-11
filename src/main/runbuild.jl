function run_build(filenamein::String; MPIparallel=false)
    # When using MPI we make sure that only rank 0 prints to the console
    if myrank == 0
        ext = splitext(filenamein)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    parameters = construct_params_from_toml(filenamein)
    MPI.Barrier(comm)

    if myrank == 0
        oneinst = parameters.numinstances == 1
        @assert MPIparallel ⊻ oneinst "MPI must be enabled if and only if numinstances > 1, numinstances was $(parameters.numinstances)"
        @assert comm_size == parameters.numinstances "numinstances has to be equal to the number of MPI ranks"
        @assert parameters.kind_of_bias ∉ ("none", "parametric") "bias has to be \"metad\" or \"opes\" in build, was $(parameters.kind_of_bias)"
        @assert parameters.is_static == false "Bias cannot be static in build"
    end

    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed * (myrank + 1))
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    logger_io = myrank == 0 ? parameters.logdir * "/logs.txt" : devnull
    set_global_logger!(parameters.verboselevel, logger_io; tc=parameters.log_to_console)

    # print time and system info, because it looks cool I guess
    # btw, all these "@level1" calls are just for logging, level1 is always printed
    # and anything higher has to specified in the parameter file (default is level2)
    @level1("Start time: @ $(current_time())")
    buf = IOBuffer()
    InteractiveUtils.versioninfo(buf)
    versioninfo = String(take!(buf))
    @level1(versioninfo)
    @level1("Random seed is: $seed\n")

    univ = Univ(parameters; use_mpi=MPIparallel)

    run_build!(univ, parameters)
    return nothing
end

function run_build!(univ, parameters)
    U = univ.U
    updatemethod = Updatemethod(parameters, U)

    gflow = GradientFlow(
        U,
        parameters.flow_integrator,
        parameters.flow_num,
        parameters.flow_steps,
        parameters.flow_tf;
        measure_every=parameters.flow_measure_every,
    )

    additional_string = "_$(myrank+1)"

    measurements = MeasurementMethods(
        U,
        parameters.measuredir,
        parameters.measurements;
        additional_string=additional_string,
    )

    measurements_with_flow = MeasurementMethods(
        U,
        parameters.measuredir,
        parameters.measurements_with_flow;
        additional_string=additional_string,
        flow=true,
    )

    build!(parameters, univ, updatemethod, gflow, measurements, measurements_with_flow)
    return nothing
end

function build!(parameters, univ, updatemethod, gflow, measurements, measurements_with_flow)
    U = univ.U
    fermion_action = univ.fermion_actions
    bias = univ.bias
    MPI.Barrier(comm)

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
                    metro_test=false,
                    friction=0,
                )
            end
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(current_time())\n-")
        end
    end

    @level1("└ Total elapsed time:\t$(runtime_therm) [s]\n")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    MPI.Barrier(comm)

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
            CVs = MPI.Allgather(U.CV::Float64, comm)
            accepted == true && update_bias!(bias, CVs, itrj, myrank == 0)
            acceptances = MPI.Allgather(numaccepts::Float64, comm) # FIXME: should use MPI.gather?
            print_acceptance_rates(acceptances, itrj)

            calc_measurements(measurements, U, itrj)
            calc_measurements_flowed(measurements_with_flow, gflow, U, itrj)
            calc_weights(bias, U.CV, itrj)
        end
    end

    @level1("└\nTotal elapsed time:\t$(convert_seconds(runtime_all))\n@ $(current_time())")
    flush(stdout)
    # close all the I/O streams
    close(updatemethod)
    close(measurements)
    close(measurements_with_flow)
    close(bias)
    close(Output.__GlobalLogger[])
    return nothing
end
