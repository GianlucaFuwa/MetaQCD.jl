struct ParameterSet
    L::NTuple{4,Int64}
    beta::Float64
    gauge_action::String
    NC::Int64
    numtherm::Int64
    numsteps::Int64
    initial::String

    # mpi-related parameters
    numprocs_cart::NTuple{4,Int64}
    halo_width::Int64

    # dynamical fermion parameters
    fermions::Vector{Dict}
    fermion_action::String
    wilson_r::Float64
    wilson_csw::Float64
    boundary_condition::String
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

    # bias parameters
    rhostout_for_cv::Float64
    kinds_of_weights::Vector{String}
    starting_Q::Union{Nothing,Vector{Int64}}
    usebiases::Vector{String}
    biases::Vector{Dict}

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
    hmc_levels::Vector{Dict}
    hmc_integrator::String
    hmc_trajectory::Float64
    hmc_friction::Float64
    hmc_rafriction::Float64
    hmc_steps::Int64
    hmc_numsmear_gauge::Int64
    hmc_numsmear_fermion::Int64
    hmc_rhostout_gauge::Float64
    hmc_rhostout_fermion::Float64
    hmc_logging::Bool
    hb_maxit::Int64
    numheatbath::Int64
    eo::Bool # XXX: Remove?
    or_algorithm::String
    numorelax::Int64
    parity_update::Bool

    # measurements
    measurements::Vector{Dict}
    measurements_with_flow::Vector{Dict}
    flow_integrator::Vector{String}
    flow_num::Int64
    flow_tf::Float64
    flow_steps::Int64
    flow_measure_every::Union{Int64,Vector{Int64}}

    # system settings
    backend::String
    float_type::String
    verboselevel::Int64
    save_config_format::String
    save_config_every::Int64
    save_checkpoint_every::Int64
    load_config_fromfile::Bool
    load_config_format::String
    load_config_path::String
    load_checkpoint_fromfile::Bool
    load_checkpoint_path::String

    randomseed::Union{UInt64,Vector{UInt64}}
    ensemble_dir::String
    log_dir::String
    measure_dir::String
    save_config_dir::String
    bias_dir::String
    log_to_console::Bool
    overwrite::Bool
end
