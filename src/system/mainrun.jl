module Mainrun
    using Dates
    using DelimitedFiles
    using InteractiveUtils
    using ..VerbosePrint
    #using Distributed
    #using Base.Threads: @spawn, nthreads
    
    import ..AbstractMeasurementModule: measure
    import ..AbstractUpdateModule: Updatemethod, update!
    import ..Gaugefields: normalize!
    import ..MetaQCD: MeasurementMethods, calc_measurement_values
    import ..ParametersTOML: construct_params_from_toml
    import ..UniverseModule: Univ
    import ..TemperingModule: temper!
    import ..VerbosePrint: VerboseLevel, println_verbose1

    function run_sim(filenamein::String)
        filename_head = splitext(filenamein)[1]
        filename = filename_head * ".toml"
        parameters = construct_params_from_toml(filename)
        rng = parameters.randomseeds
        
        univ = Univ(parameters)
        println_verbose1(univ.verbose_print, "# ", pwd())
        numinstances = univ.numinstances

        println_verbose1(univ.verbose_print, "# ", Dates.now())
        io = IOBuffer()

        InteractiveUtils.versioninfo(io)
        versioninfo = String(take!(io))
        println_verbose1(univ.verbose_print, versioninfo)

        updatemethod = []
        for i in 1:numinstances
            push!(updatemethod, Updatemethod(parameters, univ, i))
        end

        measurements = MeasurementMethods(univ.U[1], parameters.measuredir, parameters.measurement_methods)

        calc_measurement_values(measurements, 0, univ.U[1])
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
                        rng[i],
                        univ.verbose_print,
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
                        rng[i],
                        univ.verbose_print,
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
                            rng,
                        )
                        numaccepts_temper[i-1] += ifelse(accepted, 1, 0)
                    end
                end
                #save_gaugefield(savedata, univ.U, itrj)
                for i in 1:numinstances
                    measurestrings = calc_measurement_values(measurements, itrj, univ.U[i])

                    if i == 1 
                        for st in measurestrings
                            println(univ.verbose_print.fp, st)
                        end
                    end

                end

                println_verbose1(
                    univ.verbose_print,
                    "Acceptance $itrj : $(numaccepts[1]*100/itrj) %"
                )
                flush(univ.verbose_print.fp)
            end
        end

        if parameters.meta_enabled
            bias.values = bias_mean ./ parameters.Nsteps

            open(params.biasfile,"w") do io
                writedlm(io, [bias.q_vals bias.values])
            end

            println_verbose1(
                univ.verbose_print,
                "Metapotential has been saved in file \"$(params.biasfile)\""
            )

            q_vals = readdlm(
                pwd()*"/"*params.measure_dir_secondary*"/Continuous_charge.txt",
                AbstractFloat,
                comments=true)

            weights = calc_weights(q_vals[:,2], bias)

            open(params.weightfile,"w") do io
                writedlm(io, weights)
            end

            println_verbose1(
                univ.verbose_print,
                "Weights have been saved in file \"$(params.weightfile)\""
            )
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