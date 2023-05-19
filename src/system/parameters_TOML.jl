module ParametersTOML
    using Random
    using Unicode
    using TOML

    import ..ParameterStructs: struct2dict, PrintPhysicalParameters, PrintMetaParameters
    import ..ParameterStructs: PrintSystemParameters, PrintHMCrelatedParameters
    import ..ParameterStructs: PrintSmearingParameters, PrintMeasurementParameters
    import ..SystemParameters: Params

    function set_params_value!(value_Params, values)
        d = struct2dict(values)
        pnames = fieldnames(Params)

        for (i, pname_i) in enumerate(pnames)
            if haskey(d,String(pname_i))
                if d[String(pname_i)] == "nothing" 
                    value_Params[i] = nothing
                else 
                    value_Params[i] = d[String(pname_i)]
                end
            end
        end

        return nothing
    end

    function show_parameters(parameters)
        for (key, value) in parameters

            println("[$(key)]")
            
            if key == "Measurement set"
                for (key_i, value_i) in value
                    println("[$(key_i)]")
                    display(value_i)
                    println("\t")
                end
            else
                display(value)
                println("\t")
            end

        end

        return nothing
    end

    function construct_params_from_toml(filename::String)
        parameters = TOML.parsefile(filename)
        println("inputfile: ", pwd() * "/" * filename)
        construct_params_from_toml(parameters)
    end

    function construct_params_from_toml(parameters)

        show_parameters(parameters)

        pnames = fieldnames(Params)
        numparams = length(pnames)
        value_Params = Vector{Any}(undef, numparams)

        physical = PrintPhysicalParameters()
        set_params_value!(value_Params, physical)
        meta = PrintMetaParameters()
        set_params_value!(value_Params, meta)
        system = PrintSystemParameters()
        set_params_value!(value_Params, system)
        hmc = PrintHMCrelatedParameters()
        set_params_value!(value_Params, hmc)
        meas = PrintMeasurementParameters()
        set_params_value!(value_Params, meas)
        smearing = PrintSmearingParameters()
        set_params_value!(value_Params, smearing)

        #pos = findfirst(x -> String(x) == "ITERATION_MAX", pnames)
        #value_Params[pos] = 10^5

        pos = findfirst(x -> String(x) == "load_fp", pnames)
        logfilename = parameters["System Settings"]["logfile"]
        logdir = parameters["System Settings"]["logdir"]

        if isdir(logdir) == false
            mkpath(logdir)
        end

        logfile = pwd() * "/" * logdir * "/" * logfilename

        load_fp = open(logfile, "w")
        value_Params[pos] = load_fp

        measurement_basedir = parameters["System Settings"]["measurement_basedir"]
        measurement_dir = parameters["System Settings"]["measurement_dir"]

        if isdir(measurement_basedir) == false
            mkpath(measurement_basedir)
        end

        if isdir(pwd() * "/" * measurement_basedir * "/" * measurement_dir) == false
            mkpath(pwd() * "/" * measurement_basedir * "/" * measurement_dir)
        end

        pos = findfirst(x -> String(x) == "measuredir", pnames)
        measuredir = pwd() * "/" * measurement_basedir * "/" * measurement_dir
        value_Params[pos] = measuredir

        for (i, pname_i) in enumerate(pnames)
            for (key, value) in parameters
                if haskey(value, String(pname_i))
                    if String(pname_i) == "measurement_methods"
                        valuedir = construct_measurement_dir(value[String(pname_i)])
                        value_Params[i] = valuedir
                    elseif String(pname_i) == "measurements_for_smearing"
                        valuedir = construct_measurement_dir(value[String(pname_i)])
                        value_Params[i] = valuedir
                    elseif String(pname_i) == "L"
                        value_Params[i] = Tuple(value[String(pname_i)])
                    elseif String(pname_i) == "CVlims"
                        value_Params[i] = Tuple(value[String(pname_i)])
                    elseif String(pname_i) == "randomseeds"
                        value_Params[i] = eval(Meta.parse(value[String(pname_i)]))
                    elseif String(pname_i) == "kind_of_gaction"
                        value_Params[i] = Unicode.normalize(
                            value[String(pname_i)], 
                            casefold = true
                            )
                    else
                        if value[String(pname_i)] == "nothing"
                            value_Params[i] = nothing
                        else
                            value_Params[i] = value[String(pname_i)]
                        end
                    end
                end
            end

            if isassigned(value_Params, i) == false
                @error "$(pname_i) is not defined!"
            end

        end

        parameters = Params(value_Params...)

        parameter_check(parameters)

        return parameters
    end

    function parameter_check(p::Params)
        if p.saveU_format !== nothing
            if isdir(p.saveU_dir) == false
                mkpath(p.saveU_dir)
            end

            println("$(p.saveU_dir) is used for saving configurations")
        end

        if p.meta_enabled == true
            println("Metadynamics is enabled")

            if p.tempering_enabled == true
                println("Parallel tempering is enabled")
            end
        else
            println("Metadynamics is disabled")
        end

        if Unicode.normalize(p.update_method, casefold = true) == "hmc"
            println("HMC will be used")
        elseif Unicode.normalize(p.update_method, casefold = true) == "metropolis"
            println("Metropolis updates will be used")
        elseif Unicode.normalize(p.update_method, casefold = true) == "heatbath"
            println("Heatbath + Overrelaxation updates will be used")
        elseif Unicode.normalize(p.update_method, casefold = true) == "hb"
            println("Heatbath updates will be used")
        elseif Unicode.normalize(p.update_method, casefold = true) == "or"
            println("Overrelaxation updates will be used")
        else
            error("""
            update_method in [\"Physical Settings\"] = $(p.update_method) is not supported.
            Supported methods are:
            HMC
            Metropolis
            Heatbath
            """)
        end

        logdir = p.logdir

        if isdir(logdir) == false
            mkpath(logdir)
        end
    end

    function construct_measurement_dir(x)
        valuedic = Dict[]

        for (method, methoddic) in x
            dic_i = Dict()
            for (key, value) in methoddic
                    dic_i[key] = value
            end
            push!(valuedic, dic_i)
        end
        
        return valuedic
    end
end