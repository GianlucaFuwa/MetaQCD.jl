struct ParameterSet
    L::NTuple{4,Int64}
    beta::Float64
    gauge_action::String
    NC::Int64
    numtherm::Int64
    numsteps::Int64
    initial::String

    # dynamical fermion parameters
    fermion_action::String
    Nf::Union{Int,Vector{Int}}
    mass::Union{Float64,Vector{Float64}}
    wilson_r::Float64
    wilson_csw::Float64
    anti_periodic::Bool
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    rhmc_spectral_bound::NTuple{2,Float64}
    rhmc_recalc_spectral_bound::Bool
    rhmc_order_action::Int64
    rhmc_order_md::Int64
    rhmc_prec_action::Int64
    rhmc_prec_md::Int64
    eo_precon::Bool

    # bias parameters
    kind_of_bias::String
    kind_of_cv::String
    numsmears_for_cv::Int64
    rhostout_for_cv::Float64
    is_static::Union{Bool,Vector{Bool}}
    symmetric::Bool
    stride::Int64
    cvlims::NTuple{2,Float64}
    biasfactor::Float64
    kinds_of_weights::Vector{String}
    usebiases::Vector{String}
    write_bias_every::Int64
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
    hmc_numsmear_gauge::Int64
    hmc_numsmear_fermion::Int64
    hmc_rhostout_gauge::Float64
    hmc_rhostout_fermion::Float64
    hmc_logging::Bool
    hb_maxit::Int64
    numheatbath::Int64
    eo::Bool
    or_algorithm::String
    numorelax::Int64
    parity_update::Bool

    # measurements
    measurements::Vector{Dict}
    measurements_with_flow::Vector{Dict}
    flow_integrator::String
    flow_num::Int64
    flow_tf::Float64
    flow_steps::Int64
    flow_measure_every::Union{Int64,Vector{Int64}}

    # system settings
    backend::String
    float_type::String
    verboselevel::Int64
    saveU_format::String
    saveU_every::Int64
    saveU_dir::String
    loadU_format::String
    loadU_dir::String
    loadU_fromfile::Bool
    loadU_filename::String

    randomseed::Union{UInt64,Vector{UInt64}}
    ensembledir::String
    logdir::String
    log_to_console::Bool
    measuredir::String
    biasdir::String
    overwrite::Bool
end
