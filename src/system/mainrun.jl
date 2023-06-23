module Mainrun
    using Dates
    using DelimitedFiles
    using InteractiveUtils
    using Random
    using ..VerbosePrint

    import ..AbstractMeasurementModule: measure
    import ..AbstractSmearingModule: GradientFlow, Euler, RK2, RK3, RK3W7
    import ..AbstractUpdateModule: HeatbathUpdate, HMCUpdate, MetroUpdate, Updatemethod
    import ..AbstractUpdateModule: AbstractUpdate, update!
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTreeGaugeAction, SymanzikTadGaugeAction, WilsonGaugeAction
    import ..Metadynamics: MetaDisabled, MetaEnabled, calc_weights, update_bias!
    import ..MetaQCD: MeasurementMethods, calc_measurements, calc_measurements_flowed
    import ..ParametersTOML: construct_params_from_toml
    import ..UniverseModule: Univ
    import ..TemperingModule: temper!
    import ..VerbosePrint: VerboseLevel, println_verbose1

    function run_sim(filenamein::String)
        filename_head = splitext(filenamein)[1]
        filename = filename_head * ".toml"
        parameters = construct_params_from_toml(filename)

        Random.seed!(parameters.randomseed)

        univ = Univ(parameters)
        println_verbose1(univ.verbose_print, "# ", pwd())
        println_verbose1(univ.verbose_print, "# ", Dates.now())
        io = IOBuffer()

        InteractiveUtils.versioninfo(io)
        versioninfo = String(take!(io))
        println_verbose1(univ.verbose_print, versioninfo)

        println("\t>> Universe is set!\n")

        run_sim!(univ, parameters)

        return nothing
    end

    function run_sim!(univ, parameters)
        U = univ.U

        updatemethod = Vector{AbstractUpdate}(undef, univ.numinstances)
        updatemethod[1] = Updatemethod(parameters, U[1])


        for i in 2:univ.numinstances
            updatemethod[i] = HMCUpdate(
                U[i],
                parameters.hmc_integrator,
                parameters.hmc_steps,
                parameters.hmc_Δτ,
                meta_enabled = true,
            )
        end

        println("\t>> Updatemethods are set!\n")

        gradient_flow = GradientFlow(
            U[1],
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            ϵ = parameters.flow_ϵ,
            measure_every = parameters.flow_measure_every,
        )

        measurements = Vector{MeasurementMethods}(undef, parameters.numinstances)
        measurements[1] = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurement_methods,
            cv = parameters.meta_enabled,
        )

        for i in 2:parameters.numinstances
            measurements[i] = MeasurementMethods(
                U[i],
                parameters.measuredir,
                Dict[],
                cv = parameters.meta_enabled,
                additional_string = "_$i"
            )
        end

        measurements_with_flow = MeasurementMethods(
            U[1],
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
        )

        println("\t>> Measurement methods are set!\n")

        # savedata = Savedata(
        #     parameters.saveU_format,
        #     parameters.saveU_dir,
        #     parameters.saveU_every,
        #     parameters.update_method,
        #     univ.U,
        # )

        metaqcd!(
            parameters,
            univ,
            updatemethod,
            gradient_flow,
            measurements,
            measurements_with_flow,
        )

        return nothing
    end

    function metaqcd!(
        parameters,
        univ,
        updatemethod,
        gradient_flow,
        measurements,
        measurements_with_flow,
    )
        numinstances = parameters.numinstances

        U = univ.U
        Bias = univ.Bias

        calc_measurements(measurements[1], 0, U[1])

        value, runtime_therm = @timed begin
            for itrj in 1:parameters.Ntherm
                println_verbose1(univ.verbose_print, "# therm itrj = $itrj")
                updatetime = 0.0

                _, updatetime = @timed begin
                    for i in 1:numinstances
                        update!(
                            updatemethod[i],
                            U[i],
                            univ.verbose_print,
                            Bias = Bias[i],
                            metro_test = false,
                        )
                    end
                end

                println_verbose1(
                    univ.verbose_print,
                    ">> Thermalization Update: Elapsed time $(updatetime) [s]\n"
                )
            end
        end

        println_verbose1(
            univ.verbose_print,
            "\t>> Thermalization Elapsed time $(runtime_therm) [s]\n"
        )

        value, runtime_all = @timed begin
            numaccepts = zeros(numinstances)
            numaccepts_temper = numinstances > 1 ? zeros(Int64, numinstances - 1) : nothing

            for itrj in 1:parameters.Nsteps
                println_verbose1(univ.verbose_print, "\n# itrj = $itrj")

                _, updatetime = @timed begin
                    for i in 1:numinstances
                        accepted = update!(
                            updatemethod[i],
                            U[i],
                            univ.verbose_print,
                            Bias = Bias[i],
                            metro_test = true,
                        )
                        Bias[i] !== nothing ? update_bias!(Bias[i], U[i].CV) : nothing
                        numaccepts[i] += accepted
                    end
                end

                println_verbose1(
                    univ.verbose_print,
                    ">> Update: Elapsed time $(sum(updatetime)) [s]"
                )

                for i in 1:numinstances
                    println_verbose1(
                        univ.verbose_print,
                        ">> Acceptance rank_$i $itrj:\t",
                        numaccepts[i] * 100 / itrj,
                        "%",
                    )
                end

                if parameters.tempering_enabled
                    swap_every = parameters.swap_every
                    if itrj % swap_every == 0
                        for i in numinstances:-1:2
                            accepted = temper!(
                                U[i],
                                U[i-1],
                                Bias[i],
                                Bias[i-1],
                                univ.verbose_print
                            )
                            numaccepts_temper[i-1] += accepted

                            println_verbose1(
                                univ.verbose_print,
                                ">> Swap Acceptance [$i ⇔  $(i-1)] $itrj:\t",
                                "$(numaccepts_temper[i-1] * 100 / (itrj / swap_every)) %"
                            )
                        end
                    end
                end

                #save_gaugefield(savedata, univ.U, itrj)

                for i in 1:numinstances
                    calc_measurements(
                        measurements[i],
                        itrj,
                        U[i],
                    )

                    if i == 1
                        calc_measurements_flowed(
                            measurements_with_flow,
                            gradient_flow,
                            itrj,
                            U[1],
                        )
                    end

                    calc_weights(Bias[i], U[i].CV, itrj)
                end

                flush(univ.verbose_print.fp)
            end
        end

        if parameters.meta_enabled
            for i in 1:numinstances
                writedlm(
                    Bias[i].fp,
                    [Bias[i].bin_vals Bias[i].values],
                )

                close(Bias[i].fp)

                println_verbose1(
                    univ.verbose_print,
                    "\t>> Metapotential $i has been saved in file \"$(Bias[i].fp)\""
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
            end
        end

        flush(stdout)
        flush(univ.verbose_print)

        println_verbose1(univ.verbose_print, "\t>> Total Elapsed time $(runtime_all) [s]\n")

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
