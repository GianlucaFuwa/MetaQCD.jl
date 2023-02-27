module Parameters_TOML
    using TOML

    import ..Parameters: Params
    import ..Parameter_structs: 
        Print_physical_parameters,
        Print_metad_parameters,
        Print_sim_parameters,
        Print_mc_parameters,
        Print_meas_parameters,
        Print_system_parameters


function set_params_value!(value_Params,values)
    d = struct2dict(values)
    pnames = fieldnames(Params)
    for (i,pname_i) in enumerate(pnames)
        if haskey(d,String(pname_i))
            if d[String(pname_i)] == "nothing" 
                value_Params[i] = nothing
            else 
                value_Params[i] = d[String(pname_i)]
            end
        end
    end
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
end

function construct_Params_from_TOML(filename::String)
    parameters = TOML.parsefile(filename)
    println("inputfile: ", pwd() * "/" * filename)
    construct_Params_from_TOML(parameters)
end

function construct_Params_from_TOML(parameters)

    show_parameters(parameters)



    pnames = fieldnames(Params)
    numparams = length(pnames)
    value_Params = Vector{Any}(undef, numparams)

    physical = Print_physical_parameters()
    set_params_value!(value_Params, physical)
    metad = Print_metad_parameters()
    set_params_value!(value_Params, metad)
    sim = Print_sim_control_parameters()
    set_params_value!(value_Params, sim)
    mc = Print_mc_parameters()
    set_params_value!(value_Params, mc)
    meas = Print_meas_parameters()
    set_params_value!(value_Params, meas)
    system = Print_meas_parameters()
    set_params_value!(value_Params, system)

    pos = findfirst(x -> String(x) == "Nsweeps", pnames)
    value_Params[pos] = 10^5

    pos = findfirst(x -> String(x) == "load_fp", pnames)
    logfilename = parameters["System Settings"]["logfile"]
    logdir = parameters["System Settings"]["log_dir"]
    if isdir(log_dir) == false
        mkdir(log_dir)
    end
    logfile = pwd() * "/" * logdir * "/" * logfilename

    loadfp = open(logfile, "w")
    value_Params[pos] = loadfp

    measurement_dir = parameters["System Settings"]["measurement_dir"]
    if isdir(measurement_dir) == false
        mkdir(measurement_dir)
    end

    pos = findfirst(x -> String(x) == "measuredir", pnames)
    measuredir = pwd() * "/" * measurement_basedir * "/" * measurement_dir
    value_Params[pos] = measuredir

    for (i, pname_i) in enumerate(pnames)
        #println("before $(value_Params[i])")
        for (key, value) in parameters
            if haskey(value, String(pname_i))
                #println("$pname_i $key ",value[String(pname_i)])
                if String(pname_i) == "measurement_methods"
                    #println("$pname_i $key ",value[String(pname_i)])
                    valuedir = construct_measurement_dir(value[String(pname_i)])
                    value_Params[i] = valuedir
                elseif String(pname_i) == "L"
                    value_Params[i] = Tuple(value[String(pname_i)])
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

        #println(pname_i,"::$(typeof(value_Params[i])) $(value_Params[i])")

    end

    parameters = Params(value_Params...)

    parameter_check(parameters)

    return parameters
end

function parameter_check(p::Params)
    if p.Dirac_operator != nothing
        println("$(p.Dirac_operator) fermion is used")

        if p.smearing_for_fermion == "stout"
            println("stout smearing is used for fermions")
        end
    end

    if p.saveU_format ≠ nothing
        if isdir(p.saveU_dir) == false
            mkdir(p.saveU_dir)
        end
        println("$(p.saveU_dir) is used for saving configurations")
    end

    if p.update_method == "HMC"
        println("HMC will be used")
    elseif p.update_method == "HMC-MetaD"
        println("HMC-MetaD will be used")
    elseif p.update_method == "Local"
        println("Local will be used")
    elseif p.update_method == "Local-MetaD"
        println("Local-MetaD will be used")
    else
        error("""
        update_method in [\"Physical Settings\"] = $(p.update_method) is not supported.
        Supported methods are 
        HMC(-MetaD)
        Local(-MetaD)
        """)
    end

    logdir = p.logdir
    if isdir(logdir) == false
        mkdir(logdir)
    end
end

function construct_measurement_dir(x)
    valuedic = Dict[]

    for (method, methoddic) in x
        dic_i = Dict()
        for (key, value) in methoddic
                dic_i[key] = value
        end
        #println("dic_i $dic_i")
        push!(valuedic, dic_i)
    end

    return valuedic
end

end