module SystemParameters
    import ..ParameterStructs: printlist_physical, printlist_meta, printlist_system
    import ..ParameterStructs: printlist_HMCrelated, printlist_measurement

    export Params

    const printlists = [
        printlist_physical,
        printlist_meta,
        printlist_system,
        printlist_HMCrelated,
        printlist_measurement,
    ]

    #system = default_system()
    #defaultmeasures = default_defaultmeasures()
    #measurement = default_measurement()

    struct Params
        L::NTuple{4,Int64}
		β::Float64
        kind_of_gaction::String
        NC::Int64
        Ntherm::Int64
		Nsteps::Int64
		initial::String

        meta_enabled::Bool
        kind_of_cv::Union{Nothing, String}
		numsmears_for_cv::Union{Nothing, Int64}
		ρstout_for_cv::Union{Nothing, Float64}
        symmetric::Union{Nothing, Bool}
		CVlims::Union{Nothing, NTuple{2,Float64}}
		bin_width::Union{Nothing, Float64}
		meta_weight::Union{Nothing, Float64}
		penalty_weight::Union{Nothing, Float64}
        is_static::Union{Nothing, Bool, Vector{Bool}}

        tempering_enabled::Union{Nothing, Bool}
        numinstances::Int64
        swap_every::Union{Nothing, Int64}

        update_method::String
		ϵ_metro::Union{Nothing, Float64}
        multi_hit::Union{Nothing, Int64}
        metro_target_acc::Union{Nothing, Float64}
        integrator::Union{Nothing, String}
        Δτ::Union{Nothing, Float64}
        hmc_steps::Union{Nothing, Int64}
        eo::Union{Nothing, Bool}
        MAXIT::Union{Nothing, Int64}
        numHB::Union{Nothing, Int64}
        numOR::Union{Nothing, Int64}

        measurement_methods::Vector{Dict}
        smearingtype::Union{Nothing,String}
        numsmear::Union{Nothing, Int64}
        ρ_stout::Union{Nothing, Float64}

        verboselevel::Int64

        saveU_format::Union{String, Nothing}
        saveU_every::Int64
        saveU_dir::String
        loadU_format::Union{String, Nothing}
        loadU_dir::String
        loadU_fromfile::Bool
        loadU_filename::String

		randomseed::Union{Nothing, Int64}
		logdir::String
		logfile::String
		load_fp::IOStream
        measuredir::String
		savebias_dir::Union{Nothing, String, Vector{String}}
        biasfiles::Union{Nothing, String, Vector{Union{Nothing, String}}}
        usebiases::Union{Nothing, String, Vector{Union{Nothing, String}}}
        weightfiles::Union{Nothing, String, Vector{String}}
    end

end