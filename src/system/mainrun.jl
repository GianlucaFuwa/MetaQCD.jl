module Mainrun
    using Dates
    using DelimitedFiles
    using InteractiveUtils
    using Random
    using ..VerbosePrint
    
    import ..AbstractMeasurementModule: measure
    import ..AbstractSmearingModule: GradientFlow
    import ..AbstractUpdateModule: AbstractUpdate, Updatemethod, update!
    import ..Gaugefields: normalize!
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
        meta_enabled = parameters.meta_enabled
        numinstances = univ.numinstances

        println_verbose1(univ.verbose_print, "# ", Dates.now())
        io = IOBuffer()

        InteractiveUtils.versioninfo(io)
        versioninfo = String(take!(io))
        println_verbose1(univ.verbose_print, versioninfo)


        updatemethod = Vector{AbstractUpdate}(undef, numinstances)


        for i in 1:numinstances
            updatemethod[i] = Updatemethod(parameters, univ, i)
        end

        gradient_flow = GradientFlow(
            univ.U[1],
            integrator = parameters.flow_integrator,
            numflow = parameters.flow_num,
            steps = parameters.flow_steps,
            ϵ = parameters.flow_ϵ,
            measure_every = parameters.flow_measure_every,
        )

        measurements = MeasurementMethods(
            univ.U[1],
            parameters.measuredir,
            parameters.measurement_methods,
        )
        measurements_with_flow = MeasurementMethods(
            univ.U[1],
            parameters.measuredir,
            parameters.measurements_with_flow,
            flow = true,
        )

        if meta_enabled
            meta_charge_fp = Vector{IOStream}(undef, numinstances)
            for i in 1:numinstances
                meta_charge_fp[i] = open(
                    parameters.measuredir * "/Meta_charge_$i.txt",
                    "w",
                )
            end
        else
            meta_charge_fp = nothing
        end


        calc_measurements(measurements, 0, univ.U[1])

        #=
        savedata = Savedata(
            parameters.saveU_format,
            parameters.saveU_dir,
            parameters.saveU_every,
            parameters.update_method,
            univ.U,
        )
        =#
        value, runtime_therm = @timed begin
            for itrj in 1:parameters.Ntherm
                println_verbose1(univ.verbose_print, "# therm itrj = $itrj")
                updatetime = 0.0

                for i in 1:numinstances
                    _, runtime = @timed update!(
                        updatemethod[i],
                        univ.U[i],
                        univ.verbose_print,
                        Bias = univ.Bias[i],
                        metro_test = false,
                    )
                    updatetime += runtime
                    normalize!(univ.U[i])
                end

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
            numaccepts = zeros(AbstractFloat, numinstances)

            for itrj in 1:parameters.Nsteps
                println_verbose1(univ.verbose_print, "# itrj = $itrj")
                updatetime = 0.0
                
                for i in 1:numinstances
                    accepted, runtime = @timed update!(
                        updatemethod[i],
                        univ.U[i],
                        univ.verbose_print,
                        Bias = univ.Bias[i],
                    )
                    numaccepts[i] += accepted
                    updatetime += runtime
                    normalize!(univ.U[i])
                end

                println_verbose1(
                    univ.verbose_print,
                    "Update: Elapsed time $updatetime [s]"
                )
                
                if parameters.tempering_enabled
                    for i in numinstances:-1:2
                        accepted = temper!(
                            univ.U[i],
                            univ.U[i-1],
                            univ.Bias[i],
                            univ.Bias[i-1],
                        )
                        numaccepts_temper[i-1] += ifelse(accepted, 1, 0)
                    end
                end

                #save_gaugefield(savedata, univ.U, itrj)

                calc_measurements(
                    measurements,
                    itrj,
                    univ.U[1],
                )

                calc_measurements_flowed(
                    measurements_with_flow,
                    gradient_flow,
                    itrj,
                    univ.U[1],
                )

                if meta_enabled
                    for i in 1:numinstances
                        println(
                            meta_charge_fp[i],
                            "$itrj $(univ.U[i].CV) # metacharge",
                        )
                        flush(meta_charge_fp[i])
                    end
                end

                println_verbose1(
                    univ.verbose_print,
                    "Acceptance $itrj : $(numaccepts[1]*100/itrj) %"
                )
                flush(univ.verbose_print.fp)
            end
        end

        if meta_enabled

            for i in 1:numinstances
                writedlm(
                    univ.Bias[i].fp,
                    [univ.Bias[i].bin_vals univ.Bias[i].values],
                )

                close(univ.Bias[i].fp)

                println_verbose1(
                    univ.verbose_print,
                    "Metapotential $i has been saved in file \"$(univ.Bias[i].fp)\""
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

        println_verbose1(univ.verbose_print, "Total Elapsed time $(runtime_all) [s]")

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