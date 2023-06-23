module ParametersTOML
    using Unicode
    using TOML

    import ..ParameterStructs: struct2dict, PrintPhysicalParameters, PrintMetaParameters
    import ..ParameterStructs: PrintSystemParameters, PrintHMCrelatedParameters
    import ..ParameterStructs: PrintGradientFlowParameters, PrintMeasurementParameters
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

    function construct_params_from_toml(filename::String; show_params = true)
        parameters = TOML.parsefile(filename)
        println("inputfile: ", pwd() * "/" * filename)
        construct_params_from_toml(parameters, show_params = show_params)
    end

    function construct_params_from_toml(parameters; show_params = true)

        show_params ? show_parameters(parameters) : nothing

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
        gradientflow = PrintGradientFlowParameters()
        set_params_value!(value_Params, gradientflow)

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

        try
            meta_enabled = parameters["Physical Settings"]["meta_enabled"]
        catch
            @warn "Metadynamics disabled because not specified"
        end

        if @isdefined meta_enabled
        else
            meta_enabled = false
        end

        pos = findfirst(x -> String(x) == "biasdir", pnames)

        if meta_enabled == true
            bias_basedir = parameters["System Settings"]["bias_basedir"]
            bias_dir = parameters["System Settings"]["bias_dir"]

            if isdir(bias_basedir) == false
                mkpath(bias_basedir)
            end

            if isdir(pwd() * "/" * bias_basedir * "/" * bias_dir) == false
                mkpath(pwd() * "/" * bias_basedir * "/" * bias_dir)
            end

            biasdir = pwd() * "/" * bias_basedir * "/" * bias_dir
            value_Params[pos] = biasdir
        else
            biasdir = nothing
            value_Params[pos] = biasdir
        end

        for (i, pname_i) in enumerate(pnames)
            for (key, value) in parameters
                if haskey(value, String(pname_i))
                    if String(pname_i) == "measurement_methods"
                        valuedir = construct_measurement_dir(value[String(pname_i)])
                        value_Params[i] = valuedir
                    elseif String(pname_i) == "measurements_with_flow"
                        valuedir = construct_measurement_dir(value[String(pname_i)])
                        value_Params[i] = valuedir
                    elseif String(pname_i) == "L"
                        value_Params[i] = Tuple(value[String(pname_i)])
                    elseif String(pname_i) == "CVlims"
                        value_Params[i] = Tuple(value[String(pname_i)])
                    elseif String(pname_i) == "wt_factor"
                        val = value[String(pname_i)]
                        num = val == "Inf" ? Inf : val
                        @assert typeof(num)<:Real && num > 0 "wt_factor must be in (0,Inf]"
                        value_Params[i] = num
                    elseif String(pname_i) == "kind_of_gaction"
                        value_Params[i] = Unicode.normalize(
                            value[String(pname_i)],
                            casefold = true,
                        )
                    elseif String(pname_i) == "update_method"
                        value_Params[i] = Unicode.normalize(
                            value[String(pname_i)],
                            casefold = true,
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

            println("\t>> $(p.saveU_dir) is used for saving configurations\n")
        end

        if Unicode.normalize(p.update_method, casefold = true) == "hmc"
            println("\t>> HMC will be used\n")
        elseif Unicode.normalize(p.update_method, casefold = true) == "metropolis"
            println("\t>> Metropolis updates will be used\n")
        elseif Unicode.normalize(p.update_method, casefold = true) == "heatbath"
            println("\t>> Heatbath (+ Overrelaxation) updates will be used\n")
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
