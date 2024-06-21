module Parameters

using Dates
using MPI
using Unicode
using TOML

include("./parameter_structs.jl")
include("./parameter_set.jl")

const COMM = MPI.COMM_WORLD
const MYRANK = MPI.Comm_rank(COMM)

export ParameterSet

lower_case(str) = Unicode.normalize(str; casefold=true)

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
    inputfile = pwd() * "/" * filename
    MYRANK == 0 && println("inputfile: ", inputfile * "\n")
    return construct_params_from_toml(
        parameters, inputfile; am_rank0=(MYRANK == 0), backend=backend
    )
end

function construct_params_from_toml(parameters, inputfile; am_rank0=true, backend="cpu")
    pnames = fieldnames(ParameterSet)
    numparams = length(pnames)
    value_Params = Vector{Any}(undef, numparams)
    generated_dirname = Dates.format(now(), "YYYY-mm-dd-HH_MM_SS_ss")
    # generated_dirname = generate_dirname(parameters)
    # @show generated_dirname

    physical = PhysicalParameters()
    set_params_value!(value_Params, physical)
    fermion = DynamicalFermionParameters()
    set_params_value!(value_Params, fermion)
    bias = BiasParameters()
    set_params_value!(value_Params, bias)
    system = SystemParameters()
    set_params_value!(value_Params, system)
    hmc = HMCParameters()
    set_params_value!(value_Params, hmc)
    meas = MeasurementParameters()
    set_params_value!(value_Params, meas)
    gradientflow = GradientFlowParameters()
    set_params_value!(value_Params, gradientflow)

    overwrite = try
        parameters["System Settings"]["overwrite"]
    catch
        @warn "\"overwrite\" not specified in System Settings; default to true."
        true
    end

    ensembledir = try
        parameters["System Settings"]["ensemble_dir"]
    catch
        "/data/MetaQCD/$(generated_dirname)"
    end

    if !overwrite
        i = 1
        tmp = ensembledir
        while isdir(tmp) && !overwrite
            tmp = ensembledir * "_$(i)"
            i += 1
            i > 100 && error("ensemble directory name gen timed out, try \"overwrite = true\"")
        end
        ensembledir = tmp
        am_rank0 && mkpath(ensembledir)
    else
        if !isdir(ensembledir) && am_rank0
            mkpath(ensembledir)
        end
    end

    am_rank0 && cp(inputfile, ensembledir * "/used_parameterfile.toml"; force=true)
    pose = findfirst(x -> String(x) == "ensembledir", pnames)
    value_Params[pose] = ensembledir

    logdir = ensembledir * "/logs/"
    measuredir = ensembledir * "/measurements/"
    saveU_dir = ensembledir * "/configs/"
    if !isdir(logdir) && am_rank0
        mkpath(logdir)
    end
    if !isdir(measuredir) && am_rank0
        mkpath(measuredir)
    end
    if !isdir(saveU_dir) && am_rank0
        mkpath(saveU_dir)
    end
    
    posl = findfirst(x -> String(x) == "logdir", pnames)
    posm = findfirst(x -> String(x) == "measuredir", pnames)
    poss = findfirst(x -> String(x) == "saveU_dir", pnames)
    value_Params[posl] = logdir
    value_Params[posm] = measuredir
    value_Params[poss] = saveU_dir

    kind_of_bias = try
        parameters["Bias Settings"]["kind_of_bias"]
    catch
        "none"
    end

    pos = findfirst(x -> String(x) == "biasdir", pnames)

    if kind_of_bias != "none" || kind_of_bias != "parametric"
        biasdir = ensembledir * "/metapotentials/"

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
                elseif String(pname_i) == "rhmc_spectral_bound"
                    value_Params[i] = Tuple(value[String(pname_i)])
                elseif String(pname_i) == "cvlims"
                    value_Params[i] = Tuple(value[String(pname_i)])
                elseif String(pname_i) == "biasfactor"
                    val = value[String(pname_i)]
                    num = val == "Inf" ? Inf : val
                    @assert typeof(num) <: Real && num > 1 "wt_factor must be in (1,Inf]"
                    value_Params[i] = num
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

    parameter_check(parameters, am_rank0)
    MPI.Barrier(MPI.COMM_WORLD)
    return parameters
end

function parameter_check(p::ParameterSet, am_rank0)
    am_rank0 || return nothing

    if lower_case(p.gauge_action) ∉ ["wilson", "iwasaki", "symanzik_tree", "dbw2"]
        ga = p.gauge_action
        throw(AssertionError("""
              gauge_action in [\"Physical Settings\"] = $(ga) is not supported.
              Supported gactions are:
                Wilson
                Iwasaki
                DBW2
                Symanzik_tree
              """))
    end

    if lower_case(p.fermion_action) ∉ ["none", "wilson", "staggered"]
        fa = p.fermion_action
        throw(AssertionError("""
              fermion_action in [\"Physical Settings\"] = $(fa) is not supported.
              Supported gactions are:
                None
                Wilson
                Staggered
              """))
    else
        @assert lower_case(p.update_method) == "hmc" "Dynamical fermions only with HMC"
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
        um = p.update_method
        throw(AssertionError("""
            update_method in [\"Physical Settings\"] = $(um) is not supported.
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

    if lower_case(p.fermion_action) ∉ ["none", "wilson", "staggered"]
        fa = p.fermion_action
        throw(AssertionError("""
            fermion_action in [\"Dynamical Fermion Settings\"] = $(fa) is not supported.
            Supported fermion actions are:
                None
                Wilson
                Staggered
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
        fi = p.flow_integrator
        throw(AssertionError("""
             flow_integrator in [\"Gradient Flow Settings\"] = $(fi) is not supported.
             Supported methods are:
                 Euler
                 RK2
                 RK3
                 RK3W7
             """))
    end

    if lower_case(p.saveU_format) ∉ ["", "bmw", "bridge", "jld", "jld2"]
        throw(AssertionError("""
            saveU_format in [\"System Settings\"] = $(p.saveU_format) is not supported.
            Supported methods are:
                Bridge
                JLD or JLD2 (both use JLD2)
                BMW
            """))
    elseif p.saveU_format != ""
        if p.saveU_dir == ""
            !isdir(pwd() * "/configs_$(now())") && mkdir("/configs_$(now())")
        else
            !isdir(p.saveU_dir) && mkdir(p.saveU_dir)
        end
    end

    if p.loadU_fromfile
        @assert isfile(p.loadU_dir * "/" * p.loadU_filename) "Your loadU_file doesn't exist"
        if lower_case(p.loadU_format) ∉ ["bmw", "bridge", "jld", "jld2"]
            throw(AssertionError("""
            loadU_format in [\"System Settings\"] = $(p.loadU_format) is not supported.
            Supported methods are:
                Bridge
                JLD or JLD2 (both use JLD2)
                BMW
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
