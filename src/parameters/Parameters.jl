module Parameters

using Dates
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
            value_Params[i] = d[String(pname_i)]
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
    myrank==0 && println("inputfile: ", pwd() * "/" * filename * "\n")
    construct_params_from_toml(parameters, am_rank0 = (myrank==0))
end

function construct_params_from_toml(parameters; am_rank0=true)
    # am_rank0 ? save_parameters(fp, parameters) : nothing

    pnames = fieldnames(ParameterSet)
    numparams = length(pnames)
    value_Params = Vector{Any}(undef, numparams)
    time_now = now()

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

    overwrite = try
                    parameters["System Settings"]["overwrite"]
                catch
                    @warn "\"overwrite\" not specified in System Settings; default to true."
                    true
                end

    posl = findfirst(x -> String(x) == "logdir", pnames)
    log_dir = try
                  parameters["System Settings"]["log_dir"]
              catch
                  string(time_now)
              end

    logdir = pwd() * "/logs/" * log_dir

    if !isdir(logdir)
        am_rank0 && mkpath(logdir)
    end

    logfile = logdir * "/logs.txt"

    if isfile(logfile)
        overwrite || overwrite_detected("logfile", am_rank0)
    end
    value_Params[posl] = logdir

    measurement_dir = try
                          parameters["System Settings"]["measurement_dir"]
                      catch
                          string(time_now)
                      end

    measuredir = pwd() * "/measurements/" * measurement_dir
    if !isdir(measuredir)
        am_rank0 && mkpath(measuredir)
    else
        overwrite || overwrite_detected("measurement", am_rank0)
    end

    posm = findfirst(x -> String(x) == "measuredir", pnames)
    value_Params[posm] = measuredir

    kind_of_bias = try
                       parameters["Bias Settings"]["kind_of_bias"]
                   catch
                       @warn "Bias disabled because not specified"
                       "none"
                   end

    pos = findfirst(x -> String(x) == "biasdir", pnames)

    if kind_of_bias != "none"
        bias_dir = try parameters["System Settings"]["bias_dir"] catch _ "$time_now" end
        biasdir = pwd() * "/metapotentials/" * bias_dir

        if !isdir(biasdir)
            am_rank0 && mkpath(biasdir)
        end

        value_Params[pos] = biasdir
    else
        biasdir = ""
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
                elseif String(pname_i) == "biasfactor"
                    val = value[String(pname_i)]
                    num = val == "Inf" ? Inf : val
                    @assert typeof(num)<:Real && num > 1 "wt_factor must be in (1,Inf]"
                    value_Params[i] = num
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
                    value_Params[i] = value[String(pname_i)]
                end
            end
        end

        if isassigned(value_Params, i) == false
            @error "$(pname_i) is not defined!"
        end

    end

    parameters = ParameterSet(value_Params...)

    parameter_check(parameters, Val{am_rank0}())
    MPI.Barrier(MPI.COMM_WORLD)

    return parameters
end

parameter_check(::ParameterSet, ::Val{false}) = nothing

function parameter_check(p::ParameterSet, ::Val{true})
    lower_case(str) = Unicode.normalize(str, casefold=true)
    if lower_case(p.kind_of_gaction) ∉ ["wilson", "iwasaki", "symanzik_tree", "dbw2"]
        throw(AssertionError("""
              kind_of_gaction in [\"Physical Settings\"] = $(p.kind_of_gaction) is not supported.
              Supported gactions are:
                Wilson
                Iwasaki
                DBW2
                Symanzik_tree
              """))
    end

    if lower_case(p.initial) ∉ ["cold", "hot"]
        throw(AssertionError("""
            intial in [\"Physical Settings\"] = $(p.initial) is not supported.
            Supported initial conditions are:
                cold
                hot
            """))
    end

    if lower_case(p.update_method) ∉ ["hmc", "metropolis", "heatbath"]
        throw(AssertionError("""
            update_method in [\"Physical Settings\"] = $(p.update_method) is not supported.
            Supported methods are:
                HMC
                Metropolis
                Heatbath
            """))
    end

    if lower_case(p.hmc_integrator) ∉ ["leapfrog", "omf2slow", "omf2", "omf4slow", "omf4"]
        throw(AssertionError("""
            hmc_integrator in [\"HMC Settings\"] = $(p.hmc_integrator) is not supported.
            Supported methods are:
                Leapfrog
                OMF2Slow
                OMF2
                OMF4Slow
                OMF4
            """))
    end

    if lower_case(p.kind_of_bias) ∉ ["none", "metad", "metadynamics", "opes", "parametric"]
        throw(AssertionError("""
            kind_of_bias in [\"Bias Settings\"] = $(p.kind_of_bias) is not supported.
            Supported biases are:
                None
                Metadynamics/MetaD
                OPES
                Parametric
            """))
    end

    if lower_case(p.kind_of_cv) ∉ ["plaquette", "clover"]
        throw(AssertionError("""
            kind_of_cv in [\"Bias Settings\"] = $(p.kind_of_cv) is not supported.
            Supported biases are:
                Plaquette
                Clover
            """))
    end

    if lower_case(p.flow_integrator) ∉ ["euler", "rk2", "rk3", "rk3w7"]
        throw(AssertionError("""
            flow_integrator in [\"Gradient Flow Settings\"] = $(p.flow_integrator) is not supported.
            Supported methods are:
                Euler
                RK2
                RK3
                RK3W7
            """))
    end

    if lower_case(p.saveU_format) ∉ ["", "bridge", "jld", "jld2"]
        throw(AssertionError("""
            saveU_format in [\"System Settings\"] = $(p.saveU_format) is not supported.
            Supported methods are:
                Bridge
                JLD or JLD2 (both use JLD2)
            """))
    elseif p.saveU_format != ""
        if p.saveU_dir == ""
            !isdir(pwd() * "/configs_$(now())") && mkdir(pwd() * "/configs_$(now())")
        else
            !isdir(p.saveU_dir) && mkdir(p.saveU_dir)
        end
    end

    if p.loadU_fromfile
        @assert isfile(p.loadU_dir * "/" * p.loadU_filename) "Your loadU_file doesn't exist"
        if Unicode.normalize(p.loadU_format, casefold=true) ∉ ["bridge", "jld", "jld2"]
            throw(AssertionError("""
            loadU_format in [\"System Settings\"] = $(p.loadU_format) is not supported.
            Supported methods are:
                Bridge
                JLD or JLD2 (both use JLD2)
            """))
        end
    end
    return nothing
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

@noinline function overwrite_detected(s::String, am_rank0)
    am_rank0 && throw(AssertionError("""
                    The provided $s directory or file already exists
                    and \"overwrite\" in [\"System Settings\"] is set to false.
                    """))
    return nothing
end

end
