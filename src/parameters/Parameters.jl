module Parameters

using MPI
using Unicode
using TOML

include("./parameter_structs.jl")
include("./parameter_set.jl")

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

function save_parameters(fp, parameters) # TODO: Print the entirity
    for (key, value) in parameters

        println(fp, "[$(key)]")

        if key == "Measurement set"
            for (key_i, value_i) in value
                println(fp, "[$(key_i)]")
                display(value_i)
                println(fp, "\t")
            end
        else
            display(value)
            println(fp, "\t")
        end

    end

    return nothing
end

function construct_params_from_toml(filename::String)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    parameters = TOML.parsefile(filename)
    println("inputfile: ", pwd() * "/" * filename * "\n")
    construct_params_from_toml(parameters, am_rank0 = (myrank == 0))
end

function construct_params_from_toml(parameters; am_rank0 = true)
    # am_rank0 ? save_parameters(fp, parameters) : nothing

    pnames = fieldnames(ParameterSet)
    numparams = length(pnames)
    value_Params = Vector{Any}(undef, numparams)

    physical = PrintPhysicalParameters()
    set_params_value!(value_Params, physical)
    bias = PrintBiasParameters()
    set_params_value!(value_Params, bias)
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
    @assert (logfilename != "") "logdir name has to be provided"
    @assert (logdir != "") "logfile name has to be provided"

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
        kind_of_bias = parameters["Bias Settings"]["kind_of_bias"]
    catch
        @warn "Bias disabled because not specified"
    end

    if @isdefined kind_of_bias
    else
        kind_of_bias = "none"
    end

    pos = findfirst(x -> String(x) == "biasdir", pnames)

    if kind_of_bias != "none"
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
        for (_, value) in parameters
            if haskey(value, String(pname_i))
                if String(pname_i) == "measurements"
                    valuedir = construct_measurement_dicts(value[String(pname_i)])
                    value_Params[i] = valuedir
                elseif String(pname_i) == "measurements_with_flow"
                    valuedir = construct_measurement_dicts(value[String(pname_i)])
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
                elseif String(pname_i) == "biasfactor"
                    val = value[String(pname_i)]
                    num = val == "Inf" ? Inf : val
                    @assert typeof(num)<:Real && num > 1 "wt_factor must be in (1,Inf]"
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
                elseif String(pname_i) == "randomseed"
                    val = value[String(pname_i)]
                    if typeof(val) == Int64
                        value_Params[i] = UInt64(val)
                    elseif typeof(val) == Vector{String}
                        value_Params[i] = [parse(UInt64, v) for v in val]
                    else
                        value_Params[i] = val
                    end
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

    parameter_check(parameters, do_prints=am_rank0)

    return parameters
end

function parameter_check(p::ParameterSet; do_prints=true)
    if p.saveU_format !== nothing
        if isdir(p.saveU_dir) == false
            do_prints ? mkpath(p.saveU_dir) : nothing
        end

        println("\t>> configs are saved in $(p.saveU_dir)\n")
    end

    if Unicode.normalize(p.update_method, casefold=true) ∈ ["hmc", "metropolis", "heatbath"]
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

function construct_measurement_dicts(x)
    meas_dicts = Dict[]

    for (method, method_dict) in x
        dictᵢ = Dict()
        dictᵢ["observable"] = method

        for (key, value) in method_dict
            dictᵢ[key] = value
        end

        push!(meas_dicts, dictᵢ)
    end

    return meas_dicts
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
