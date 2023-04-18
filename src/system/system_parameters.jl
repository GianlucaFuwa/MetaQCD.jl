module System_parameters
    using Random
    export Params
    import ..Parameter_structs:
        printlist_physical,
        printlist_meta,
        printlist_system,
        printlist_HMCrelated,
        printlist_measurement

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
        NC::Int64
        Ntherm::Int64
		Nsweeps::Int64
		initial::String

        meta_enabled::Bool
        symmetric::Union{Nothing,Bool}
		CVlims::Union{Nothing,NTuple{2,Float64}}
		bin_width::Union{Nothing,Float64}
		w::Union{Nothing,Float64}
		k::Union{Nothing,Float64}
        is_static::Union{Nothing,Vector{Bool}}

        tempering_enabled::Union{Nothing,Bool}
        numinstances::Union{Nothing,Int64}
        swap_every::Union{Nothing,Int64}

        update_method::String
		ϵ_local::Union{Nothing,Float64}
        integrator::Union{Nothing,String}
        ϵ_hmc::Union{Nothing,Float64}
        hmc_steps::Union{Nothing,Int64}

        measurement_methods::Vector{Dict}
        measure_every::Int64
        smearing_type::Union{Nothing,String}
        numsmear::Union{Nothing,Int64}
        ρ_stout::Union{Nothing,Float64}

        verboselevel::Int64

		randomseeds::Union{Xoshiro,Vector{Xoshiro}}
		logdir::String
		logfile::String
		loadfile::IOStream
        measure_basedir::String
		measure_dir::String
		savebias_dir::Union{Nothing,String,Vector{String}}
        biasfiles::Union{Nothing,String,Vector{Union{Nothing,String}}}
        usebiases::Union{Nothing,String,Vector{Union{Nothing,String}}}
        weightfiles::Union{Nothing,String,Vector{String}}
    end

end