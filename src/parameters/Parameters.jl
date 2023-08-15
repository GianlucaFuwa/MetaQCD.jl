module Parameters
    using MPI
    using Unicode
    using TOML

    include("./parameter_structs.jl")
    include("./parameter_set.jl")
    include("../measurements/measurement_methods.jl")

    export ParameterSet

    function set_params_value!(value_Params, values)
        d = struct2dict(values)
        pnames = fieldnames(ParameterSet)

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
        myrank = MPI.Comm_rank(MPI.COMM_WORLD)
        parameters = TOML.parsefile(filename)
        println("inputfile: ", pwd() * "/" * filename)
        construct_params_from_toml(parameters, am_rank0 = (myrank == 0))
    end

    function construct_params_from_toml(parameters; am_rank0 = true)
        am_rank0 ? show_parameters(parameters) : nothing

        pnames = fieldnames(ParameterSet)
        numparams = length(pnames)
        value_Params = Vector{Any}(undef, numparams)

        physical = PrintPhysicalParameters()
        set_params_value!(value_Params, physical)
        meta = PrintMetaParameters()
        set_params_value!(value_Params, meta)
        system = PrintSystemParameters()
        set_params_value!(value_Params, system)
        hmc = PrintHMCParameters()
        set_params_value!(value_Params, hmc)
        meas = PrintMeasurementParameters()
        set_params_value!(value_Params, meas)
        gradientflow = PrintGradientFlowParameters()
        set_params_value!(value_Params, gradientflow)

        overwrite = true

        try
            overwrite = parameters["System Settings"]["overwrite"]
        catch
            @warn "\"overwrite\" not specified in System Settings; default to true."
        end

        pos = findfirst(x -> String(x) == "load_fp", pnames)
        logfilename = parameters["System Settings"]["logfile"]
        logdir = parameters["System Settings"]["logdir"]

        if isdir(logdir) == false
            am_rank0 ? mkpath(logdir) : nothing
        end

        logfile = pwd() * "/" * logdir * "/" * logfilename

        if isfile(logfile)
            overwrite ? nothing : overwrite_detected("logfile", am_rank0)
        end

        load_fp = am_rank0 ? open(logfile, "w") : nothing
        value_Params[pos] = load_fp

        measurement_basedir = parameters["System Settings"]["measurement_basedir"]
        measurement_dir = parameters["System Settings"]["measurement_dir"]

        if isdir(measurement_basedir) == false
            am_rank0 ? mkpath(measurement_basedir) : nothing
        end

        if isdir(pwd() * "/" * measurement_basedir * "/" * measurement_dir) == false
            if am_rank0
                mkpath(pwd() * "/" * measurement_basedir * "/" * measurement_dir)
            end
        else
            overwrite ? nothing : overwrite_detected("measurement", am_rank0)
        end

        pos = findfirst(x -> String(x) == "measuredir", pnames)
        measuredir = pwd() * "/" * measurement_basedir * "/" * measurement_dir
        value_Params[pos] = measuredir

        try
            meta_enabled = parameters["MetaD Settings"]["meta_enabled"]
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
                am_rank0 ? mkpath(bias_basedir) : nothing
            end

            if isdir(pwd() * "/" * bias_basedir * "/" * bias_dir) == false
                am_rank0 ? mkpath(pwd() * "/" * bias_basedir * "/" * bias_dir) : nothing
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
                    elseif String(pname_i) == "cvlims"
                        value_Params[i] = Tuple(value[String(pname_i)])
                    elseif String(pname_i) == "usebiases"
                        value_Params[i] = Vector{Union{Nothing, String}}(
                            value[String(pname_i)]
                        )
                        for (idx, entry) in enumerate(value_Params[i])
                            if entry == "nothing"
                                value_Params[i][idx] = nothing
                            end
                        end
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

        parameters = ParameterSet(value_Params...)

        parameter_check(parameters, do_prints = am_rank0)

        return parameters
    end

    function parameter_check(p::ParameterSet; do_prints = true)
        if p.saveU_format !== nothing
            if isdir(p.saveU_dir) == false
                do_prints ? mkpath(p.saveU_dir) : nothing
            end

            println("\t>> configs are saved in $(p.saveU_dir)\n")
        end

        if Unicode.normalize(p.update_method, casefold = true) == "hmc"
            do_prints ? println("\t>> HMC will be used\n") : nothing
        elseif Unicode.normalize(p.update_method, casefold = true) == "metropolis"
            do_prints ? println("\t>> Metropolis updates will be used\n") : nothing
        elseif Unicode.normalize(p.update_method, casefold = true) == "heatbath"
            do_prints ? println("\t>> Heatbath (+ Overrelaxation) updates will be used\n") : nothing
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
            do_prints ? mkpath(logdir) : nothing
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

    function overwrite_detected(s::String, am_rank0)
        am_rank0 || return nothing

        println(
            ">> The provided $s directory or file already exists",
            " and files might be overwritten\n",
            ">> Do you want to continue? [y/n]"
        )

        answer = readline()

        if answer == "y"
            println(">> Execution proceeds")
        elseif answer == "n"
            println(">> Execution cancelled")
            MPI.Abort(MPI.COMM_WORLD, 1)
            exit()
        else
            overwrite_detected(s, am_rank0)
        end
    end

end
