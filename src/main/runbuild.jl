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
        @assert mpi_size() == parameters.numinstances """
        numinstances has to be equal to the number of MPI ranks
        """
        @assert multi_sim ⊻ oneinst """
        MPI must be enabled only if numinstances > 1 or fields are distributed
        numinstances was: $(parameters.numinstances) but comm size was $(mpi_size())
        """
        @assert parameters.kind_of_bias ∉ ("none", "parametric") """
        bias has to be \"metad\" or \"opes\" in build, was $(parameters.kind_of_bias)
        """
    end

    rid = mpi_myrank()+1
    @assert parameters.is_static[rid] == false "Bias $rid cannot be static in build"
    @assert length(parameters.usebiases) <= 1 "Only one prebuilt bias can be parsed in build"
    if length(parameters.usebiases) == 1
        for _ in 1:mpi_size()-1
            push!(parameters.usebiases, parameters.usebiases[1])
        end
    end

    # set random seed if provided, otherwise generate one
    if parameters.randomseed != 0
        seed = parameters.randomseed
        Random.seed!(seed * (mpi_myrank() + 1) % UInt64)
    else
        seed = rand(UInt64)
        Random.seed!(seed)
    end

    logpath = joinpath(parameters.log_dir, "logs_$(lpad(mpi_myrank(), 3, "0")).txt")
    to_console = mpi_amroot() ? parameters.log_to_console : false
    set_global_logger!(
        parameters.verboselevel, logpath; tc=to_console
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
    therm_cv = Vector{Float64}(undef, parameters.numtherm)
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

            if adaptive_σ
                recalc_CV!(U, bias)
                therm_cv[itrj] = U.CV
            end
            @level1("|  Elapsed time:\t$(updatetime) [s] @ $(string(current_time()))")
        end
    end

    @level1("- Thermalization elapsed time:\t$(runtime_therm) [s]\n")
    recalc_CV!(U, bias) # need to recalc cv since it was not updated during therm

    mpi_barrier()

    if adaptive_σ
        std_cv = mpi_allgather(std(therm_cv)::Float64, comm)
        set_σ₀!(bias, mean(std_cv))
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
            CVs = mpi_allgather(U.CV::Float64, comm)
            bval = mpi_allgather(CVs[1]::Float64, mpi_comm())
            @assert all(x -> x==bval[1], bval) "Biases are not properly synchronized"
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
