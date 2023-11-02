struct ParameterSet
    L::NTuple{4,Int64}
    beta::Float64
    kind_of_gaction::String
    NC::Int64
    numtherm::Int64
    numsteps::Int64
    initial::String

    # bias parameters
    kind_of_bias::String
    kind_of_cv::String
    numsmears_for_cv::Int64
    rhostout_for_cv::Float64
    is_static::Union{Bool, Vector{Bool}}
    symmetric::Bool
    stride::Int64
    cvlims::NTuple{2,Float64}
    biasfactor::Float64
    kinds_of_weights::Vector{String}
    usebiases::Union{Nothing, Vector{Union{Nothing, String}}}
    write_bias_every::Union{Nothing, Int64}
    # metadynamics specific parameters
    bin_width::Float64
    meta_weight::Float64
    penalty_weight::Float64
    # opes specific parameters
    barrier::Float64
    sigma0::Float64
    sigma_min::Float64
    fixed_sigma::Bool
    no_Z::Bool
    opes_epsilon::Float64
    threshold::Float64
    cutoff::Float64
    # for parametric
    bias_Q::Float64
    bias_A::Float64
    bias_Z::Float64

    # tempering parameters
    tempering_enabled::Bool
    numinstances::Int64
    swap_every::Int64
    non_metadynamics_updates::Int64
    measure_on_all::Bool

    # update parameters
    update_method::String
    metro_epsilon::Float64
    metro_numhits::Int64
    metro_target_acc::Float64
    hmc_integrator::String
    hmc_trajectory::Float64
    hmc_friction::Float64
    hmc_steps::Int64
    hmc_numsmear::Int64
    hmc_rhostout::Float64
    hb_maxit::Int64
    numheatbath::Int64
    eo::Bool
    or_algorithm::String
    numorelax::Int64
    parity_update::Bool

    measurement_methods::Vector{Dict}
    measurements_with_flow::Vector{Dict}
    flow_integrator::String
    flow_num::Int64
    flow_tf::Float64
    flow_steps::Int64
    flow_measure_every::Union{Nothing, Int64, Vector{Int64}}

    verboselevel::Int64

    saveU_format::Union{String, Nothing}
    saveU_every::Int64
    saveU_dir::String
    loadU_format::Union{String, Nothing}
    loadU_dir::String
    loadU_fromfile::Bool
    loadU_filename::String

    randomseed::Union{UInt64, Vector{UInt64}}
    logdir::String
    logfile::String
    load_fp::Union{Nothing, IOStream}
    measuredir::String
    biasdir::Union{Nothing, String}
    overwrite::Bool
end
