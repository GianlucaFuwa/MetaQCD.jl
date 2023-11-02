function run_build(filenamein::String; MPIparallel=false)
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)

    if myrank == 0
        println("\t>> MPI enabled with $(MPI.Comm_size(comm)) procs\n")
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

    updatemethod = Updatemethod(parameters, U)

    gradient_flow = GradientFlow(
        U,
        integrator = parameters.flow_integrator,
        numflow = parameters.flow_num,
        steps = parameters.flow_steps,
        tf = parameters.flow_tf,
        measure_every = parameters.flow_measure_every,
    )

    additional_string = "_$(MPI.Comm_rank(MPI.COMM_WORLD))"

    measurements = MeasurementMethods(
        U,
        parameters.measuredir,
        parameters.measurement_methods,
        cv = true,
        additional_string = additional_string,
    )
    measurements_with_flow = MeasurementMethods(
        U,
        parameters.measuredir,
        parameters.measurements_with_flow,
        flow = true,
    )

    build!(
        parameters,
        univ,
        updatemethod,
        gradient_flow,
        measurements,
        measurements_with_flow,
    )

    return nothing
end

function build!(
    parameters,
    univ,
    update_method,
    gradient_flow,
    measurements,
    measurements_with_flow,
)
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    U = univ.U
    bias = univ.bias
    vp = univ.verbose_print

    calc_measurements(measurements, U, 0)

    MPI.Barrier(comm)

    _, runtime_therm = @timed begin
        for itrj in 1:parameters.numtherm
            println_verbose0(vp, "\n# therm itrj = $itrj")

            _, updatetime = @timed begin
                update!(update_method, U, vp, bias=nothing, metro_test=false)
            end

            println_verbose0(vp, "Thermalization Update: Elapsed time $(updatetime) [s]")
        end
    end

    MPI.Barrier(comm)

    println_verbose0(vp, "Thermalization Elapsed time $(runtime_therm) [s]")

    _, runtime_all = @timed begin
        numaccepts = 0.0

        for itrj in 1:parameters.numsteps
            println_verbose0(vp, "\n# itrj = $itrj")

            _, updatetime = @timed begin
                numaccepts += update!(update_method, U, vp, bias=bias)
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
            calc_measurements_flowed(measurements_with_flow, gradient_flow, U, itrj)
            calc_weights(bias, U.CV, itrj)
        end
    end

    if myrank == 0
        println_verbose1(vp, "\n\t>> Total Elapsed time $(runtime_all) [s]")
        flush(univ.verbose_print)
        flush(stdout)
    end

    MPI.Finalize()
    return nothing
end
