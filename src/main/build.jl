function run_build(filenamein::String; MPIparallel=false)
    if myrank == 0
        MPIparallel && println("\t>> MPI enabled with $(comm_size) procs\n")
        ext = splitext(filenamein)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    parameters = construct_params_from_toml(filenamein)
    MPI.Barrier(comm)

    @assert parameters.kind_of_bias != "none" "bias has to be enabled in build"
    @assert parameters.is_static==false "Bias $(myrank+1) cannot be static in build"

    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed * (myrank + 1))
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    logger_io = myrank==0 ? parameters.logdir*"/logs.txt" : devnull
    set_global_logger!(parameters.verboselevel, logger_io, tc=parameters.log_to_console)
    @level1("Start time: @ $(current_time())")
    buf = IOBuffer()
    InteractiveUtils.versioninfo(buf)
    versioninfo = String(take!(buf))
    @level1(versioninfo)
    @level1("Random seed is: $seed\n")

    univ = Univ(parameters, use_mpi=MPIparallel)

    run_build!(univ, parameters)
    return nothing
end

function run_build!(univ, parameters)
    U = univ.U
    updatemethod = Updatemethod(parameters, U)

    gflow = GradientFlow(U, parameters.flow_integrator, parameters.flow_num,
                         parameters.flow_steps, parameters.flow_tf,
                         measure_every = parameters.flow_measure_every)

    additional_string = "_$(myrank+1)"

    measurements = MeasurementMethods(U, parameters.measuredir, parameters.measurements,
                                      additional_string = additional_string)

    measurements_with_flow = MeasurementMethods(U, parameters.measuredir,
                                                parameters.measurements_with_flow,
                                                additional_string = additional_string,
                                                flow = true)

    build!(parameters, univ, updatemethod, gflow, measurements, measurements_with_flow)
    return nothing
end

function build!(parameters,univ, updatemethod, gflow, measurements, measurements_with_flow)
    U = univ.U
    bias = univ.bias

    @level1("┌ Thermalization:")
    _, runtime_therm = @timed begin
        for itrj in 1:parameters.numtherm
            @level1("|  itrj = $itrj")
            _, updatetime = @timed begin
                update!(updatemethod, U, bias=nothing, metro_test=false, friction=π/2)
            end
            @level1("|  Elapsed time:\t$(updatetime) [s]")
        end
    end

    @level1("└ Total elapsed time:\t$(runtime_therm) [s]\n")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    MPI.Barrier(comm)

    @level1("┌ Production:")
    _, runtime_all = @timed begin
        numaccepts = 0.0

        for itrj in 1:parameters.numsteps
            @level1("|  itrj = $itrj")

            _, updatetime = @timed begin
                accepted = update!(updatemethod, U, bias=bias)
                numaccepts += accepted
            end

            @level1("|  Elapsed time:\t$(updatetime) [s]\n")

            CVs = MPI.Allgather(U.CV::Float64, comm)
            accepted==true && update_bias!(bias, CVs, itrj, myrank==0)
            acceptances = MPI.Allgather(numaccepts::Float64, comm)
            print_acceptance_rates(acceptances, itrj)

            calc_measurements(measurements, U, itrj)
            calc_measurements_flowed(measurements_with_flow, gflow, U, itrj)
            calc_weights(bias, U.CV, itrj)
        end
    end

    @level1("└\n" *
            "Total elapsed time:\t$(convert_seconds(runtime_all))\n" *
            "@ $(current_time())")
    flush(stdout)
    close(updatemethod)
    close(measurements)
    close(measurements_with_flow)
    close(bias)
    close(Output.GlobalLogger[])
    return nothing
end
