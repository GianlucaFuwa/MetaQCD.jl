module Mainbuild
    using Dates
    using DelimitedFiles
    using InteractiveUtils
    using MPI
    using Random
    using ..VerbosePrint

    import ..AbstractMeasurementModule: measure
    import ..AbstractSmearingModule: GradientFlow, Euler, RK2, RK3, RK3W7
    import ..AbstractUpdateModule: HeatbathUpdate, HMCUpdate, MetroUpdate, Updatemethod
    import ..AbstractUpdateModule: update!
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTreeGaugeAction, SymanzikTadGaugeAction, WilsonGaugeAction
    import ..Gaugefields: normalize!
    import ..Metadynamics: MetaDisabled, MetaEnabled, update_bias!
    import ..MetaQCD: MeasurementMethods, calc_measurements, calc_measurements_flowed
    import ..ParametersTOML: construct_params_from_toml
    import ..UniverseModule: Univ
    import ..TemperingModule: temper!
    import ..VerbosePrint: VerboseLevel, println_verbose1

    function run_build(filenamein::String; MPIparallel = false)
        if MPIparallel == true
            comm = MPI.COMM_WORLD

            if MPI.Comm_rank(comm) == 0
                show_params = true
                println("\t>> MPI enabled with $(MPI.Comm_size(comm)) procs\n")
            else
                show_params = false
            end
        else
            show_params = true
        end

        ext = splitext(filenamein)[end]
        @assert (ext == ".toml") "input file format \"$ext\" not supported. Use TOML format"

        parameters = construct_params_from_toml(filenamein; show_params = show_params)

        @assert parameters.meta_enabled == true "meta_enabled has to be true in build"

        for (idx, val) in enumerate(parameters.is_static)
            (val == true) && @warn "Stream $idx is static, which is probably not wanted"
        end

        if MPIparallel == true
            fp = (MPI.Comm_rank(MPI.COMM_WORLD) == 0)
            Random.seed!(parameters.randomseed * MPI.Comm_rank(MPI.COMM_WORLD))
        else
            fp = true
            Random.seed!(parameters.randomseed)
        end

        univ = Univ(parameters, use_mpi = MPIparallel, fp = fp)
        println_verbose1(univ.verbose_print, "# ", pwd())

        println_verbose1(univ.verbose_print, "# ", Dates.now())
        io = IOBuffer()

        InteractiveUtils.versioninfo(io)
        versioninfo = String(take!(io))
        println_verbose1(univ.verbose_print, versioninfo)

        run_build!(univ, parameters, MPIparallel = MPIparallel)

        return nothing
    end

    function run_build!(univ, parameters; MPIparallel = false)
        U = univ.U

        tmp = Updatemethod(parameters, U[1])
        updatemethod = Vector{typeof(tmp)}(undef, univ.numinstances)
        updatemethod[1] = tmp


        for i in 2:univ.numinstances
            updatemethod[i] = Updatemethod(parameters, U[i])
        end

        gradient_flow = GradientFlow(
            U[1],
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            ϵ = parameters.flow_ϵ,
            measure_every = parameters.flow_measure_every,
        )

        measurements = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurement_methods,
        )
        measurements_with_flow = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
        )

        # savedata = Savedata(
        #     parameters.saveU_format,
        #     parameters.saveU_dir,
        #     parameters.saveU_every,
        #     parameters.update_method,
        #     univ.U,
        # )

        if MPIparallel == true
            build_parallel!(
                parameters,
                univ,
                updatemethod,
                gradient_flow,
                measurements,
                measurements_with_flow,
            )
        else
            build!(
                parameters,
                univ,
                updatemethod,
                gradient_flow,
                measurements,
                measurements_with_flow,
            )
        end

        return nothing
    end

    function build!(
        parameters,
        univ,
        updatemethod,
        gradient_flow,
        measurements,
        measurements_with_flow,
    )
        U = univ.U[1]
        Bias = univ.Bias[1]
        updatem = updatemethod[1]

        meta_charge_fp = open(
            parameters.measuredir * "/Meta_charge.txt",
        )

        calc_measurements(measurements, 0, U)

        value, runtime_therm = @timed begin
            for itrj in 1:parameters.Ntherm
                println_verbose1(univ.verbose_print, "# therm itrj = $itrj")

                _, updatetime = @timed update!(
                    updatem,
                    U,
                    univ.verbose_print,
                    metro_test = false,
                )
                updatetime += runtime
                normalize!(U)

                println_verbose1(
                    univ.verbose_print,
                    "Thermalization Update: Elapsed time $updatetime [s]"
                )
            end
        end

        println_verbose1(
            univ.verbose_print,
            "Thermalization Elapsed time $(runtime_therm) [s]"
        )

        value, runtime_all = @timed begin
            numaccepts = 0.0

            for itrj in 1:parameters.Nsteps
                println_verbose1(univ.verbose_print, "# itrj = $itrj")

                accepted, updatetime = @timed update!(
                    updatem,
                    U,
                    univ.verbose_print,
                    Bias = Bias,
                )
                numaccepts += accepted
                normalize!(U)

                println_verbose1(
                    univ.verbose_print,
                    "Update: Elapsed time $updatetime [s]"
                )

                #save_gaugefield(savedata, univ.U, itrj)

                calc_measurements(
                    measurements,
                    itrj,
                    U,
                )

                calc_measurements_flowed(
                    measurements_with_flow,
                    gradient_flow,
                    itrj,
                    U,
                )

                println(
                    meta_charge_fp,
                    "$itrj $(U.CV) # metacharge",
                )
                flush(meta_charge_fp)

                println_verbose1(
                    univ.verbose_print,
                    "Acceptance $itrj : $(numaccepts*100/itrj) %"
                )
                flush(univ.verbose_print.fp)
            end
        end

        writedlm(
            Bias.fp,
            [Bias.bin_vals Bias.values],
        )

        close(univ.Bias.fp)

        println_verbose1(
            univ.verbose_print,
            "Metapotential has been saved in file \"$(Bias.fp)\""
        )
        #=
        q_vals = readdlm(
            parameters.measuredir * "/Meta_charge_$i.txt",
            Float64,
            comments = true,
        )

        weights = calc_weights(q_vals[:,2], univ.Bias[i])

        open(parameters.measuredir * "/Weights_$i.txt", "w") do io
            writedlm(io, weights)
        end

        println_verbose1(
            univ.verbose_print,
            "Weights $i have been saved"
        )
        =#

        flush(stdout)
        flush(univ.verbose_print)

        println_verbose1(univ.verbose_print, "Total Elapsed time $(runtime_all) [s]")

        return nothing
    end

    function build_parallel!(
        parameters,
        univ,
        updatemethod,
        gradient_flow,
        measurements,
        measurements_with_flow,
    )
        comm = MPI.COMM_WORLD
        myrank = MPI.Comm_rank(comm)
        U = univ.U[1]
        Bias = univ.Bias[1]
        updatem = updatemethod[1]

        meta_charge_fp = Verbose1(open(
            parameters.measuredir * "/Meta_charge_$(myrank + 1).txt",
            "w",
        ))

        if myrank == 0
            calc_measurements(measurements, 0, U)
        end

        MPI.Barrier(comm)

        value, runtime_therm = @timed begin
            for itrj in 1:parameters.Ntherm
                if myrank == 0
                    println_verbose1(univ.verbose_print, "# therm itrj = $itrj")
                end

                _, updatetime = @timed update!(
                    updatem,
                    U,
                    univ.verbose_print,
                    Bias = Bias,
                    metro_test = false,
                )
                normalize!(U)

                if myrank == 0
                    println_verbose1(
                        univ.verbose_print,
                        "Thermalization Update: Elapsed time $(updatetime) [s]"
                    )
                end
            end
        end

        MPI.Barrier(comm)

        if myrank == 0
            println_verbose1(
                univ.verbose_print,
                "Thermalization Elapsed time $(runtime_therm) [s]"
            )
        end

        value, runtime_all = @timed begin
            numaccepts = 0.0

            for itrj in 1:parameters.Nsteps
                if myrank == 0
                    println_verbose1(univ.verbose_print, "# itrj = $itrj")
                end

                accepted, updatetime = @timed update!(
                    updatem,
                    U,
                    univ.verbose_print,
                    Bias = Bias,
                )
                numaccepts += accepted
                normalize!(U)

                MPI.Barrier(comm)

                CVs = MPI.Allgather(U.CV, comm)

                for cv in CVs
                    update_bias!(Bias, cv)
                end

                if myrank == 0
                    println_verbose1(
                        univ.verbose_print,
                        "Update: Elapsed time $(updatetime) [s]"
                    )

                    #save_gaugefield(savedata, univ.U, itrj)

                    calc_measurements(
                        measurements,
                        itrj,
                        U,
                    )

                    calc_measurements_flowed(
                        measurements_with_flow,
                        gradient_flow,
                        itrj,
                        U,
                    )
                end

                println_verbose1(
                    meta_charge_fp,
                    "$itrj\t$(U.CV)\t# metacharge_$(myrank)",
                )
                flush(meta_charge_fp)

                if myrank == 0
                    println_verbose1(
                        univ.verbose_print,
                        ">> Acceptance rank_$myrank $itrj:\t",
                        numaccepts * 100 / itrj,
                        "%",
                    )
                    flush(univ.verbose_print.fp)
                end
            end
        end

        if myrank == 0
            writedlm(
                Bias.fp,
                [Bias.bin_vals Bias.values],
            )

            println_verbose1(
                univ.verbose_print,
                "Cumulative Metapotential has been saved in file \"$(Bias.fp)\""
            )
            close(Bias.fp)
            flush(univ.verbose_print)
            flush(stdout)
            println_verbose1(univ.verbose_print, "Total Elapsed time $(runtime_all) [s]")
        end

        MPI.Finalize()

        return nothing
    end

    mutable struct SaveData
        issaved::Bool
        saveU_format::Union{Nothing, String}
        saveU_dir::String
        saveU_every::Int
        itrjsavecount::Int

        function SaveData(saveU_format, saveU_dir, saveU_every, update_method, U)
            itrjsavecount = 0

            if saveU_format !== nothing && update_method != "Fileloading"
                itrj = 0
                itrjstring = lpad(itrj, 8, "0")
                println_verbose1(U, "save gaugefields U every $(saveU_every) trajectory")
                issaved = true
            else
                issaved = false
            end

            return new(issaved, saveU_format, saveU_dir, saveU_every, itrjsavecount)
        end
    end
    #=
    function save_gaugefield(savedata::Savedata, U, itrj)
        if savedata.issaved == false
            return
        end

        if itrj % savedata.itrjsavecount == 0
            savedata.itrjsavecount += 1
            itrjstring = lpad(itrj, 8, "0")
            if savedata.saveU_format == "JLD"
                filename = savedata.saveU_dir * "/conf_$(itrjstring).jld2"
                saveU(filename, U)
            elseif savedata.saveU_format == "ILDG"
                filename = savedata.saveU_dir * "/conf_$(itrjstring).ildg"
                save_binarydata(U, filename)
            elseif savedata.saveU_format == "BridgeText"
                filename = savedata.saveU_dir * "/conf_$(itrjstring).txt"
                save_textdata(U, filename)
            else
                error("$(savedata.saveU_format) is not supported")
            end
        end
    end
    =#
end
