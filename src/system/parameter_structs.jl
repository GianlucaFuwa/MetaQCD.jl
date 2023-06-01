module ParameterStructs
    import ..AbstractMeasurementModule: MeasurementParameters
    @enum SmearingMethod NoSmearing = 1 STOUT = 2

    import ..AbstractMeasurementModule: construct_measurement_parameters_from_dict
    import ..AbstractMeasurementModule: PlaquetteParameters, PolyakovParameters
    import ..AbstractMeasurementModule: prepare_measurement_from_dict
    import ..AbstractMeasurementModule: TopologicalChargeParameters, WilsonLoopParameters

    const important_parameters = [
        "L",
        "β",
        "kind_of_gaction",
        "update_method",
        "ϵ_metro",
        "hmc_steps",
        "Δτ",
        "meta_enabled",
        "tempering_enabled",
        "kind_of_cv",
        "numCVsmear",
        "methodname",
        "measurement_basedir",
        "hassmearing",
        "measurement_dir",
        "kinds_of_topological_charge",
        "measurements_for_smearing",
        "smearing_measurements",
    ]

    function check_important_parameters(key)
        findornot = findfirst(x -> x==key, important_parameters)

        if findornot === nothing
            return false
        else 
            return true
        end

        return nothing
    end

    function struct2dict(x::T) where {T}
        dict = Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))
        return dict
    end

    function generate_printlist(x::Type)
        pnames = fieldnames(x)
        plist = String[]

        for i in eachindex(pnames)
            push!(plist, String(pnames[i]))
        end

        return plist
    end

    Base.@kwdef mutable struct PrintPhysicalParameters
        L::NTuple{4, Int64} = (4, 4, 4, 4)
        β::Float64 = 5.7
        NC::Int64 = 3
        kind_of_gaction::String = "wilson"
        Ntherm::Int64 = 10
        Nsteps::Int64 = 100
        inital::String = "cold"
        update_method::Vector{String} = ["HMC"]
        meta_enabled::Bool = false
        tempering_enabled::Bool = false
        numinstances::Int64 = 1
        swap_every::Union{Nothing, Int64} = nothing
        eo::Bool = false
        MAXIT::Int64 = 10^5
        numHB::Int64 = 4
        numOR::Int64 = 1
        ϵ_metro::Float64 = 0.1
        multi_hit::Int64 = 1
        metro_target_acc::Float64 = 0.5
    end

    Base.@kwdef mutable struct PrintMetaParameters
        meta_enabled::Bool = false
        kind_of_cv::String = "clover"
		numsmears_for_cv::Int64 = 4
		ρstout_for_cv::Float64 = 0.125
        symmetric::Bool = false
        CVlims::NTuple{2, Float64} = (-7, 7)
        bin_width::Float64 = 1e-2
        meta_weight::Float64 = 1e-3
        penalty_weight::Float64 = 1000.0
        is_static::Union{Bool, Vector{Bool}} = false
    end

    Base.@kwdef mutable struct PrintSystemParameters
        log_dir::String = ""
        logfile::String = ""
        verboselevel::Int64 = 1
        loadU_format::Union{Nothing, String} = nothing
        loadU_dir::String = ""
        loadU_fromfile::Bool = false
        loadU_filename::String = ""
        saveU_dir::String = ""
        saveU_format::Union{String, Nothing} = nothing
        saveU_every::Int64 = 1
        randomseed::Union{Nothing, Int64} = nothing
        measurement_basedir::String = ""
        measurement_dir::String = ""
        savebias_dir::Union{Nothing, String, Vector{String}} = nothing
        biasfiles::Union{Nothing, String, Vector{Union{Nothing,String}}} = nothing
        usebiases::Union{Nothing, String, Vector{Union{Nothing,String}}} = nothing
        weightfiles::Union{Nothing, String, Vector{String}} = nothing
    end

    Base.@kwdef mutable struct PrintHMCrelatedParameters
        Δτ::Float64 = 0.1
        hmc_steps::Int64 = 10
        integrator::String = "Leapfrog"
    end

    Base.@kwdef mutable struct PrintSmearingParameters
        hassmearing::Bool = false
        smearingtype::String = "Stout"
        ρ_stout::Float64 = 0.125
        numsmear::Int64 = 1
    end

    Base.@kwdef mutable struct PrintMeasurementParameters
        measurement_method::Vector{Dict} = Dict[]
    end

    const printlist_physical = generate_printlist(PrintPhysicalParameters)
    const printlist_meta = generate_printlist(PrintMetaParameters)
    const printlist_system = generate_printlist(PrintSystemParameters)
    const printlist_HMCrelated = generate_printlist(PrintHMCrelatedParameters)
    const printlist_measurement = generate_printlist(PrintMeasurementParameters)

    abstract type SmearingParameters end

    Base.@kwdef mutable struct NoSmearingParameters <: SmearingParameters end

    Base.@kwdef mutable struct MeasurementParameterSet
        measurement_methods::Vector{MeasurementParameters} = []
    end

    function transform_measurement_dictvec(value)
        smear_dict = Dict()
        nummeasure = length(value)
        value_out = Vector{Measurement_parameters}(undef, nummeasure)
        hassmearing = false

        for i in 1:nummeasure
            if haskey(value[i], "methodname")
                if value[i]["methodname"] == "Topological_charge"
                    hassmearing = true
                    value_out[i] = transform_topological_charge_measurement!(
                        smear_dict,
                        value[i],
                    )
                else
                    value_out[i] = construct_Measurement_parameters_from_dict(value[i])
                end
            else
                error("method name in measurement should be set")
            end
        end

        return value_out, smear_dict, hassmearing
    end

    function transform_topological_charge_measurement!(smear_dict, measurement)
        @assert haskey(measurement, "methodname") "method name in measurement should be set"
        @assert measurement["methodname"] == "Topological_charge" "function is for top. charge"

        measurement_revised = Dict()

        for (key,value) in measurement
            if key == "numsmear"
                flow_dict["numsmear"] = value
            elseif key == "ρ_stout"
                flow_dict["ρ_stout"] = value
            end
        end

        value_m = construct_Measurement_parameters_from_dict(measurement_revised)
        smear_dict["measurements_for_smearing"] = Dict()
        smear_dict["measurements_for_smearing"]["Topological_charge"] = measurement_revised

        return value_m
    end

    function construct_printable_parameters_fromdict!(
        key,
        value,
        physical,
        meta,
        system,
        hmc,
    )
        if key == "L"
            value = collect(value)
        elseif key == "r"
            value = Float64(value)
        end

        hasvalue = false
        pname_i = Symbol(key)
        physical_index = findfirst(x -> x==key, printlist_physical)

        if physical_index !== nothing
            setfield!(physical, pname_i, value)
            hasvalue = true
        end
        
        meta_index = findfirst(x -> x==key, printlist_meta)

        if meta_index !== nothing
            setfield!(meta, pname_i, value)
            hasvalue = true
        end

        system_index = findfirst(x -> x==key, printlist_system)

        if system_index !== nothing
            setfield!(system, pname_i, value)
            hasvalue = true
        end

        hmc_index = findfirst(x -> x==key, printlist_HMCrelated)

        if hmc_index !== nothing
            setfield!(hmc, pname_i, value)
            hasvalue = true
        end

        if hasvalue == false
            @warn "$(key) is not used"
        end

        return hasvalue
    end

    function construct_printable_parameters_fromdict!(
        x::Dict,
        physical,
        meta,
        system,
        hmc
    )
        for (key, value) in x
            hasvalue = false
            pname_i = Symbol(key)
            physical_index = findfirst(x -> x==pname_i, printlist_physical)

            if physical_index !== nothing
                setfield!(physical, pname_i, value)
                hasvalue = true
            end
            
            meta_index = findfirst(x -> x==pname_i, printlist_meta)

            if meta_index !== nothing
                setfield!(meta, pname_i, value)
                hasvalue = true
            end

            system_index = findfirst(x -> x==pname_i, printlist_system)

            if system_index !== nothing
                setfield!(system, pname_i, value)
                hasvalue = true
            end

            hmc_index = findfirst(x -> x==pname_i, printlist_HMCrelated)

            if hmc_index !== nothing
                setfield!(hmc, pname_i, value)
                hasvalue = true
            end

            if hasvalue == false
                @warn "$(pname_i) is not used"
            end
        end

        return nothing
    end

    function remove_default_values!(x::Dict, defaultsystem)
        for (key, value) in x
            if hasfield(typeof(defaultsystem), Symbol(key))
                default_value = getfield(defaultsystem,Symbol(key))

                if value == default_value || string(value) == string(default_value)
                    if check_important_parameters(key) == false
                        delete!(x, key)
                    end
                else
                    if value === nothing
                        x[key] = "nothing"
                    end
                end

            else
                if value === nothing
                    x[key] = "nothing"
                end
            end

            if typeof(value) == Vector{Measurement_parameters}
                construct_dict_from_measurement!(x, value)
            end
        end

        return nothing
    end

    function construct_dict_from_measurement!(x, value)
        measuredic = Dict()

        for measure in value
            methoddic = struct2dict(measure)
            measure_struct_default = typeof(measure)()
            remove_default_values!(methoddic, measure_struct_default)
            measuredic[methoddic["methodname"]] = methoddic
        end

        x["measurement_methods"] = measuredic
        return nothing
    end

    function remove_default_values!(x::Dict)
        physical = Print_physical_parameters()
        meta = Print_meta_parameters()
        system = Print_system_parameters()
        hmc = Print_HMCrelated_parameters()

        for (params, paramsname) in x
            remove_default_values!(x[params], physical)
            remove_default_values!(x[params], meta)
            remove_default_values!(x[params], system)
            remove_default_values!(x[params], hmc)
        end

        return nothing
    end

end