module Parameters

using Dates
using Unicode
using TOML
using ..Utils

include("./parameter_structs.jl")
include("./parameter_set.jl")

export ParameterSet

function set_params_value!(value_Params, values)
    d = struct2dict(values)
    pnames = fieldnames(ParameterSet)

    for (i, pname_i) in enumerate(pnames)
        if haskey(d, String(pname_i))
            value_Params[i] = d[String(pname_i)]
        end
    end
    return nothing
end

function save_parameters(fp, parameters) # XXX: We already create a copy of the parameter file
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

function construct_params_from_toml(filename::String; backend="cpu")
    parameters = TOML.parsefile(filename)
    inputfile = isabspath(filename) ? filename : joinpath(pwd(), filename)
    mpi_amroot() && println("inputfile: ", inputfile * "\n")
    return construct_params_from_toml(parameters, inputfile; backend=backend)
end

function construct_params_from_toml(parameters, inputfile; backend="cpu")
    pnames = fieldnames(ParameterSet)
    numparams = length(pnames)
    value_Params = Vector{Any}(undef, numparams)
    generated_dirname = Dates.format(now(), "YYYY-mm-dd-HH_MM_SS_ss")
    # generated_dirname = generate_dirname(parameters)
    # @show generated_dirname

    ensemble = EnsembleParameters()
    set_params_value!(value_Params, ensemble)
    data = DataParameters()
    set_params_value!(value_Params, data)
    gauge = GaugeActionParameters()
    set_params_value!(value_Params, gauge)
    fermion = FermionActionParameters()
    set_params_value!(value_Params, fermion)
    bias = BiasParameters()
    set_params_value!(value_Params, bias)
    hmc = HMCParameters()
    set_params_value!(value_Params, hmc)
    meas = MeasurementParameters()
    set_params_value!(value_Params, meas)
    gradientflow = GradientFlowParameters()
    set_params_value!(value_Params, gradientflow)

    overwrite = try
        parameters["Data Settings"]["overwrite"]
    catch
        @warn "\"overwrite\" not specified in System Settings; default to true."
        true
    end

    ensemble_dir = try
        parameters["Data Settings"]["ensemble_dir"]
    catch
        tmp = if mpi_amroot()
            tmpp = "$(homedir())/data/MetaQCD/$(generated_dirname)"
            @info "\"ensemble_dir\" not specified! Data will be stored in $tmpp"
            tmpp
        else
            nothing
        end

        ed = mpi_bcast_isbits(tmp)
        ed
    end

    if !overwrite
        i = 1
        tmp = ensemble_dir
        while isdir(tmp) && !overwrite
            tmp = ensemble_dir * "_$(i)"
            i += 1
            i > 100 && error("ensemble directory name gen timed out, try \"overwrite = true\"")
        end
        ensemble_dir = tmp
        mpi_amroot() && mkpath(ensemble_dir)
    else
        if !isdir(ensemble_dir) && mpi_amroot()
            mkpath(ensemble_dir)
        end
    end

    mpi_amroot() && cp(inputfile, joinpath(ensemble_dir, "used_parameterfile.toml"); force=true)
    pose = findfirst(x -> String(x) == "ensemble_dir", pnames)
    value_Params[pose] = ensemble_dir

    log_dir = joinpath(ensemble_dir, "logs/")
    measure_dir = joinpath(ensemble_dir, "measurements/")
    save_config_dir = joinpath(ensemble_dir, "configs/")
    bias_dir = joinpath(ensemble_dir, "biaspotentials/")

    if !isdir(log_dir) && mpi_amroot()
        mkpath(log_dir)
    end

    if !isdir(measure_dir) && mpi_amroot()
        mkpath(measure_dir)
    end

    if !isdir(save_config_dir) && mpi_amroot()
        mkpath(save_config_dir)
    end

    if !isdir(bias_dir) && mpi_amroot()
        mkpath(bias_dir)
    end
    
    posl = findfirst(x -> String(x) == "log_dir", pnames)
    posm = findfirst(x -> String(x) == "measure_dir", pnames)
    poss = findfirst(x -> String(x) == "save_config_dir", pnames)
    posb = findfirst(x -> String(x) == "bias_dir", pnames)
    value_Params[posl] = log_dir
    value_Params[posm] = measure_dir
    value_Params[poss] = save_config_dir
    value_Params[posb] = bias_dir

    # Make sure every process can see the new directories
    ensemble_dir_exists = isdir(ensemble_dir)
    log_dir_exists = isdir(log_dir)
    measure_dir_exists = isdir(measure_dir)
    config_dir_exists = isdir(save_config_dir)
    bias_dir_exists = isdir(bias_dir)
    itimer = 0

    while !(
        ensemble_dir_exists && log_dir_exists &&
        measure_dir_exists && config_dir_exists && bias_dir_exists
    )
        itimer == 50 && error("Rank $(mpi_myrank()) could not find all directories")
        sleep(0.1)
        ensemble_dir_exists = isdir(ensemble_dir)
        log_dir_exists = isdir(log_dir)
        measure_dir_exists = isdir(measure_dir)
        config_dir_exists = isdir(save_config_dir)
        bias_dir_exists = isdir(bias_dir)
        itimer += 1
    end

    mpi_barrier()

    for (i, pname_i) in enumerate(pnames)
        for (_, value) in parameters
            if haskey(value, String(pname_i))
                if String(pname_i) == "measurements"
                    valuedir = construct_measurement_dicts(value[String(pname_i)])
                    value_Params[i] = valuedir
                elseif String(pname_i) == "measurements_with_flow"
                    valuedir = construct_measurement_dicts(value[String(pname_i)])
                    value_Params[i] = valuedir
                elseif String(pname_i) == "biases"
                    valuedir = construct_bias_dicts(value[String(pname_i)])
                    value_Params[i] = valuedir
                elseif String(pname_i) == "L"
                    value_Params[i] = Tuple(value[String(pname_i)])
                elseif String(pname_i) == "flow_integrator"
                    if value[String(pname_i)] isa String
                        value_Params[i] = [value[String(pname_i)]]
                    else
                        value_Params[i] = value[String(pname_i)]
                    end
                elseif String(pname_i) == "rhmc_spectral_bound"
                    value_Params[i] = Tuple(value[String(pname_i)])
                elseif String(pname_i) == "randomseed"
                    val = value[String(pname_i)]
                    if typeof(val) == Int64
                        value_Params[i] = UInt64(val)
                    elseif typeof(val) == Vector{Int64}
                        value_Params[i] = [UInt64(v) for v in val]
                    elseif typeof(val) == Vector{String}
                        value_Params[i] = [parse(UInt64, v) for v in val]
                    else
                        value_Params[i] = val
                    end
                elseif String(pname_i) == "numprocs_cart"
                    value_Params[i] = Tuple(value[String(pname_i)])
                elseif String(pname_i) == "backend"
                    value_Params[i] = backend
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

    check_parameters(parameters)
    mpi_barrier()
    return parameters
end

function check_parameters(p::ParameterSet)
    mpi_amroot() || return nothing

    @assert prod(p.numprocs_cart) <= mpi_size() """
    Size of comm must equal number of process used in field decomposition if distributed
    or bigger if using multiple walkers
    """

    @assert p.verboselevel > 0 "verboselevel in parameters has to be bigger than 0"

    if prod(p.numprocs_cart) > 1
        @assert p.halo_width >= 1 "Halo width must be >= 1, when using field decomposition"
        @assert lower_case(p.update_method) == "hmc" """
        Field decomposition not supported for local update algorithms
        """
        
        if p.gauge_action != "wilson"
            @assert p.halo_width >= 2 """
            Halo width must be >= 2, when using field decomposition with improved \
            gauge action
            """
        end
    end

    @assert lower_case(p.gauge_action) ∈ ["wilson", "iwasaki", "symanzik_tree", "dbw2"] """
    gauge_action in [\"Physical Settings\"]: \"$(p.gauge_action)\" is not supported.
    Supported gactions are:
    Wilson
    Iwasaki
    DBW2
    Symanzik_tree
    """

    @assert lower_case(p.fermion_action) ∈ [
        "none", "wilson", "staggered",
        "staggered-h1234", "staggered-h1324", "staggered-h1342"
    ] """
    fermion_action in [\"Physical Settings\"]: \"$(p.fermion_action)\" is not supported.
    Supported gactions are:
    None
    Wilson
    Staggered
    Staggered-H1234
    Staggered-H1324
    Staggered-H1342
    """

    if lower_case(p.fermion_action) != "none"
        @assert lower_case(p.update_method) == "hmc" "Dynamical fermions only with HMC"
    end

    @assert lower_case(p.initial) ∈ ["cold", "hot"] """
    intial in [\"Physical Settings\"]: \"$(p.initial)\" is not supported.
    Supported initial conditions are:
    cold
    hot
    """

    @assert lower_case(p.update_method) ∈ ["hmc", "metropolis", "heatbath"] """
    update_method in [\"Physical Settings\"]: \"$(p.update_method)\" is not supported.
    Supported methods are:
    HMC
    Metropolis
    Heatbath
    """

    @assert lower_case(p.hmc_integrator) ∈ [
        "leapfrog", "omf2slow", "omf2", "omf4slow", "omf4", "leapfrogra", "omf4ra"
    ] """
    hmc_integrator in [\"HMC Settings\"]: \"$(p.hmc_integrator)\" is not supported.
    Supported methods are:
    Leapfrog
    LeapfrogRA
    OMF2Slow
    OMF2
    OMF4Slow
    OMF4
    OMF4RA
    """

    for flow_int in p.flow_integrator
        @assert lower_case(flow_int) ∈ ["euler", "rk2", "rk3", "rk3w7", "cooling"] """
        flow_integrator in [\"Gradient Flow Settings\"]: \"$(flow_int)\" is not supported.
        Supported methods are:
        Euler
        RK2
        RK3
        RK3W7
        Cooling
        """
    end

    @assert lower_case(p.save_config_format) ∈ ["", "bmw", "bridge", "jld", "jld2"] """
    save_config_format in [\"System Settings\"]: \"$(p.save_config_format)\" \
    is not supported.
    Supported methods are:
    Bridge
    JLD or JLD2 (both use JLD2)
    BMW
    """

    if p.load_config_fromfile
        @assert isfile(p.load_config_path) "Your load_config_path doesn't exist"
        @assert lower_case(p.load_config_format) ∈ ["bmw", "bridge", "jld", "jld2"] """
        loadU_format in [\"System Settings\"]: \"$(p.load_config_format)\" \
        is not supported.
        Supported methods are:
        Bridge
        JLD or JLD2 (both use JLD2)
        BMW
        """
    end

    return nothing
end

function construct_fermion_dicts(x)
    fermion_dicts = Dict[]

    for (method, method_dict) in x
        dictᵢ = Dict()
        dictᵢ["fermion"] = method

        for (key, value) in method_dict
            dictᵢ[key] = value
        end

        push!(fermion_dicts, dictᵢ)
    end

    return fermion_dicts
end

function construct_level_dicts(x)
    level_dicts = Dict[]

    for (method, method_dict) in x
        dictᵢ = Dict()
        dictᵢ["level"] = parse(Int64, method)

        for (key, value) in method_dict
            dictᵢ[key] = value
        end

        push!(level_dicts, dictᵢ)
    end

    return level_dicts
end

function construct_bias_dicts(x)
    bias_dicts = Dict[]

    for (method, method_dict) in x
        dictᵢ = Dict()
        dictᵢ["bias"] = method

        for (key, value) in method_dict
            dictᵢ[key] = value
        end

        push!(bias_dicts, dictᵢ)
    end

    return bias_dicts
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

@noinline function overwrite_detected(s::String)
    mpi_amroot() && throw(AssertionError("""
                    The provided $s directory or file already exists
                    and \"overwrite\" in [\"System Settings\"] is set to false.
                    """))
    return nothing
end

function generate_dirname(parameters)
    time_now = Dates.format(now(), "YYYY-mm-dd-HH_MM_SS_ss")
    NX, NY, NZ, NT = parameters["Physical Settings"]["L"]

    gauge_str = try
        parameters["Physical Settings"]["gauge_action"]
    catch _
        "wilson"
    end
    beta_str = try
        parameters["Physical Settings"]["beta"]
    catch
        error("beta has to defined in [\"Physical Settings\"]")
    end
    fermion_str = try
        parameters["Dynamical Fermion Settings"]["fermion_action"]
    catch _
        ""
    end
    Nf_str = if fermion_str != ""
        try
            Nf = parameters["Dynamical Fermion Settings"]["Nf"]
            str = "_Nf"
            for i in eachindex(Nf)
                i == length(Nf) && continue
                str *= "$(Nf[i])+"
            end
            str *= "$(Nf[end])"
        catch _
            error("Nf has to be defined in [\"Dynamical Fermion Settings\"]")
        end
    else
        ""
    end
    mass_str = if fermion_str != ""
        try
            mass = parameters["Dynamical Fermion Settings"]["mass"]
            str = "_mass"
            for i in eachindex(mass)
                i == length(mass) && continue
                str *= "$(mass[i])+"
            end
            str *= "$(mass[end])"
        catch _
            error("mass has to be defined in [\"Dynamical Fermion Settings\"]")
        end
    else
        ""
    end

    dirname = "$(NX)x$(NY)x$(NZ)x$(NT)_$(gauge_str)_beta$(beta_str)_$(fermion_str)$(Nf_str)$(mass_str)"
    return dirname * "_$(time_now)"
end

end
