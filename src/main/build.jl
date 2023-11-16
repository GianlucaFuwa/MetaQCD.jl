function run_build(filenamein::String; MPIparallel=false)
    if myrank == 0
        println("\t>> MPI enabled with $(comm_size) procs\n")
        ext = splitext(filenamein)[end]
        @assert (ext == ".toml") """
            input file format \"$ext\" not supported. Use TOML format
        """
    end

    parameters = construct_params_from_toml(filenamein)
    MPI.Barrier(comm)

    @assert parameters.kind_of_bias != "none" "bias has to be enabled in build"

    (parameters.is_static==true) && @warn(
        "Stream_$(myrank+1) is static, which is probably not wanted"
    )

    fp = (myrank == 0)

    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed * (myrank + 1))
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    univ = Univ(parameters, use_mpi=MPIparallel, fp=fp)

    println_verbose0(univ.verbose_print, ">> Random seed is: $seed")

    run_build!(univ, parameters)

    return nothing
end

function run_build!(univ, parameters)
    U = univ.U
    UM_verbose = (myrank==0) ? univ.verbose_print : nothing
    updatemethod = Updatemethod(parameters, U, UM_verbose)

    gflow = GradientFlow(
        U,
        integrator = parameters.flow_integrator,
        numflow = parameters.flow_num,
        steps = parameters.flow_steps,
        tf = parameters.flow_tf,
        measure_every = parameters.flow_measure_every,
        verbose = UM_verbose,
    )

    additional_string = "_$(myrank)"

    println_verbose0(univ.verbose_print, ">> Preparing Measurements...")
    measurements = MeasurementMethods(
        U,
        parameters.measuredir,
        parameters.measurements,
        additional_string = additional_string,
        verbose = UM_verbose,
    )
    println_verbose0(univ.verbose_print, ">> Preparing flowed Measurements...")
    measurements_with_flow = MeasurementMethods(
        U,
        parameters.measuredir,
        parameters.measurements_with_flow,
        additional_string = additional_string,
        flow = true,
        verbose = UM_verbose,
    )

    build!(
        parameters,
        univ,
        updatemethod,
        gflow,
        measurements,
        measurements_with_flow,
    )

    return nothing
end

function build!(
    parameters,
    univ,
    updatemethod,
    gflow,
    measurements,
    measurements_with_flow,
)
    U = univ.U
    bias = univ.bias
    vp = univ.verbose_print

    _, runtime_therm = @timed begin
        for itrj in 1:parameters.numtherm
            println_verbose0(vp, "\n# therm itrj = $itrj")

            _, updatetime = @timed begin
                update!(updatemethod, U, vp, bias=nothing, metro_test=false)
            end

            println_verbose0(vp, ">> Therm. Update elapsed time:\t$(updatetime) [s]\n#")
        end
    end

    println_verbose0(vp, "\t>> Thermalization elapsed time:\t$(runtime_therm) [s]\n")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    MPI.Barrier(comm)

    _, runtime_all = @timed begin
        numaccepts = 0.0

        for itrj in 1:parameters.numsteps
            println_verbose0(vp, "\n# itrj = $itrj")

            _, updatetime = @timed begin
                numaccepts += update!(updatemethod, U, vp, bias=bias)
            end

            println_verbose0(vp, "Update: Elapsed time $(updatetime) [s]")

            CVs = MPI.Allgather(U.CV, comm)
            update_bias!(bias, CVs, itrj, myrank==0)

            acceptances = MPI.Allgather(numaccepts, comm)

            if myrank == 0
                print_acceptance_rates(acceptances, itrj, vp)
                flush(vp)
            end

            MPI.Barrier(comm)

            calc_measurements(measurements, U, itrj)
            calc_measurements_flowed(measurements_with_flow, gflow, U, itrj)
            calc_weights(bias, U.CV, itrj)
        end
    end

    close(measurements)
    close(measurements_with_flow)
    close(bias)

    if myrank == 0
        println_verbose0(vp, "\n\t>> Total Elapsed time $(runtime_all) [s]")
        close(vp)
        flush(stdout)
    end

    return nothing
end
