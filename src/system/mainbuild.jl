module Mainbuild
    using Dates
    using DelimitedFiles
    using InteractiveUtils
    using MPI
    using Printf
    using Random
    using ..VerbosePrint

    import ..AbstractMeasurementModule: measure
    import ..AbstractSmearingModule: GradientFlow, Euler, RK2, RK3, RK3W7
    import ..AbstractUpdateModule: HeatbathUpdate, HMCUpdate, MetroUpdate, Updatemethod
    import ..AbstractUpdateModule: update!
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTreeGaugeAction, SymanzikTadGaugeAction, WilsonGaugeAction
    import ..Gaugefields: normalize!
    import ..Metadynamics: MetaDisabled, MetaEnabled, calc_weights, update_bias!
    import ..MetaQCD: MeasurementMethods, calc_measurements, calc_measurements_flowed
    import ..ParametersTOML: construct_params_from_toml
    import ..UniverseModule: Univ
    import ..TemperingModule: temper!
    import ..VerbosePrint: VerboseLevel, println_verbose1

    function run_build(filenamein::String; MPIparallel = false)
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

        @assert parameters.meta_enabled == true "meta_enabled has to be true in build"

        (parameters.is_static == true) && @warn(
            "Stream_$idx is static, which is probably not wanted"
        )

        fp = (myrank == 0)

        if parameters.randomseed != 0
            seed = parameters.randomseed
            Random.seed!(seed * (myrank + 1))
        else
            seed = rand(1:1_000_000_000)
            Random.seed!(seed)
        end

        univ = Univ(parameters, use_mpi = MPIparallel, fp = fp)

        if myrank == 0
            println_verbose1(univ.verbose_print, "# ", pwd())

            println_verbose1(univ.verbose_print, "# ", Dates.now())
            io = IOBuffer()

            InteractiveUtils.versioninfo(io)
            versioninfo = String(take!(io))
            println_verbose1(univ.verbose_print, versioninfo)
            println_verbose1(univ.verbose_print, ">> Random seed is: $seed")
        end

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
            ϵ = parameters.flow_ϵ,
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

    # function build!(
    #     parameters,
    #     univ,
    #     updatemethod,
    #     gradient_flow,
    #     measurements,
    #     measurements_with_flow,
    # )
    #     U = univ.U[1]
    #     Bias = univ.Bias[1]
    #     update_method = updatemethod[1]

    #     calc_measurements(measurements, 0, U)

    #     value, runtime_therm = @timed begin
    #         println_verbose1(univ.verbose_print, "\n# therm itrj = $itrj")

    #         _, updatetime = @timed update!(
    #             update_method,
    #             U,
    #             univ.verbose_print,
    #             metro_test = false,
    #         )
    #         normalize!(U)

    #         println_verbose1(
    #             univ.verbose_print,
    #             "Thermalization Update: Elapsed time $updatetime [s]\n"
    #         )
    #     end

    #     println_verbose1(
    #         univ.verbose_print,
    #         "Thermalization Elapsed time $(runtime_therm) [s]"
    #     )

    #     value, runtime_all = @timed begin
    #         numaccepts = 0.0

    #         for itrj in 1:parameters.Nsteps
    #             println_verbose1(univ.verbose_print, "\n# itrj = $itrj")

    #             accepted, updatetime = @timed update!(
    #                 update_method,
    #                 U,
    #                 univ.verbose_print,
    #                 Bias = Bias,
    #             )
    #             numaccepts += accepted
    #             normalize!(U)

    #             println_verbose1(
    #                 univ.verbose_print,
    #                 "Update: Elapsed time $updatetime [s]"
    #             )

    #             println_verbose1(
    #                 univ.verbose_print,
    #                 "Acceptance $itrj : $(numaccepts*100/itrj) %"
    #             )

    #             #save_gaugefield(savedata, univ.U, itrj)

    #             calc_measurements(
    #                 measurements,
    #                 itrj,
    #                 U,
    #             )

    #             calc_measurements_flowed(
    #                 measurements_with_flow,
    #                 gradient_flow,
    #                 itrj,
    #                 U,
    #             )

    #             calc_weights(Bias, U.CV, itrj)
    #             flush(univ.verbose_print.fp)
    #         end
    #     end

    #     writedlm(
    #         Bias.fp,
    #         [Bias.bin_vals Bias.values],
    #     )

    #     close(univ.Bias.fp)

    #     println_verbose1(
    #         univ.verbose_print,
    #         "Metapotential has been saved in file \"$(Bias.fp)\""
    #     )

    #     flush(stdout)
    #     flush(univ.verbose_print)

    #     println_verbose1(univ.verbose_print, "Total Elapsed time $(runtime_all) [s]")

    #     return nothing
    # end

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
        Bias = univ.Bias

        calc_measurements(measurements, 0, U)

        MPI.Barrier(comm)

        value, runtime_therm = @timed begin
            for itrj in 1:parameters.Ntherm
                if myrank == 0
                    println_verbose1(univ.verbose_print, "\n# therm itrj = $itrj")
                end

                _, updatetime = @timed update!(
                    update_method,
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
                    println_verbose1(univ.verbose_print, "\n# itrj = $itrj")
                end

                accepted, updatetime = @timed update!(
                    update_method,
                    U,
                    univ.verbose_print,
                    Bias = Bias,
                )
                numaccepts += accepted
                normalize!(U)

                if myrank == 0
                    println_verbose1(
                        univ.verbose_print,
                        "Update: Elapsed time $(updatetime) [s]"
                    )

                    #save_gaugefield(savedata, univ.U, itrj)
                end

                CVs = MPI.Allgather(U.CV, comm)

                for cv in CVs
                    update_bias!(Bias, cv)
                end

                acceptances = MPI.Allgather(numaccepts, comm)

                if myrank == 0
                    for (i, value) in enumerate(acceptances)
                        println_verbose1(
                            univ.verbose_print,
                            ">> Acceptance rank_$i $itrj:\t$(value * 100 / itrj)%",
                        )
                    end
                    flush(univ.verbose_print)
                end

                MPI.Barrier(comm)

                measurestrings = calc_measurements(
                    measurements,
                    itrj,
                    U,
                )
                measurestrings_flowed = calc_measurements_flowed(
                    measurements_with_flow,
                    gradient_flow,
                    itrj,
                    U,
                )

                if myrank == 0
                    for (i, value) in enumerate(measurestrings)
                        println(value)
                    end

                    for (i, value) in enumerate(measurestrings_flowed)
                        println(value)
                    end

                    calc_weights(Bias, U.CV, itrj)
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
                "\nCumulative Metapotential has been saved in file \"$(Bias.fp)\""
            )
            close(Bias.fp)
            println_verbose1(
                univ.verbose_print,
                "\n\t>> Total Elapsed time $(runtime_all) [s]",
            )
            flush(univ.verbose_print)
            flush(stdout)
        end

        MPI.Finalize()

        return nothing
    end
end
