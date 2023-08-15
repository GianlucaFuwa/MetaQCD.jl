const printlists = [
    printlist_physical,
    printlist_meta,
    printlist_system,
    printlist_hmc,
    printlist_measurement,
]

#system = default_system()
#defaultmeasures = default_defaultmeasures()
#measurement = default_measurement()

struct ParameterSet
    L::NTuple{4,Int64}
    beta::Float64
    kind_of_gaction::String
    NC::Int64
    numtherm::Int64
    numsteps::Int64
    initial::String

    meta_enabled::Bool
    kind_of_cv::Union{Nothing, String}
    numsmears_for_cv::Union{Nothing, Int64}
    rhostout_for_cv::Union{Nothing, Float64}
    symmetric::Union{Nothing, Bool}
    cvlims::Union{Nothing, NTuple{2,Float64}}
    bin_width::Union{Nothing, Float64}
    meta_weight::Union{Nothing, Float64}
    penalty_weight::Union{Nothing, Float64}
    wt_factor::Union{Nothing, Float64}
    is_static::Union{Nothing, Bool, Vector{Bool}}
    kinds_of_weights::Union{Nothing, Vector{String}}
    usebiases::Union{Nothing, Vector{Union{Nothing, String}}}

    tempering_enabled::Union{Nothing, Bool}
    numinstances::Int64
    swap_every::Union{Nothing, Int64}
    non_metadynamics_updates::Union{Nothing, Int64}
    measure_on_all::Union{Nothing, Bool}

    update_method::String
    metro_epsilon::Union{Nothing, Float64}
    metro_multi_hit::Union{Nothing, Int64}
    metro_target_acc::Union{Nothing, Float64}
    hmc_integrator::Union{Nothing, String}
    hmc_deltatau::Union{Nothing, Float64}
    hmc_steps::Union{Nothing, Int64}
    hmc_numsmear::Union{Nothing, Int64}
    hmc_rhostout::Union{Nothing, Float64}
    hb_maxit::Union{Nothing, Int64}
    numheatbath::Union{Nothing, Int64}
    eo::Union{Nothing, Bool}
    numorelax::Union{Nothing, Int64}
    parity_update::Bool

    measurement_methods::Vector{Dict}
    measurements_with_flow::Union{Nothing, Vector{Dict}}
    flow_integrator::Union{Nothing, String}
    flow_num::Union{Nothing, Int64}
    flow_tf::Union{Nothing, Float64}
    flow_steps::Union{Nothing, Int64}
    flow_measure_every::Union{Nothing, Int64}

    verboselevel::Int64

    saveU_format::Union{String, Nothing}
    saveU_every::Int64
    saveU_dir::String
    loadU_format::Union{String, Nothing}
    loadU_dir::String
    loadU_fromfile::Bool
    loadU_filename::String

    randomseed::Int64
    logdir::String
    logfile::String
    load_fp::Union{Nothing, IOStream}
    measuredir::String
    biasdir::Union{Nothing, String}
    overwrite::Bool
end
