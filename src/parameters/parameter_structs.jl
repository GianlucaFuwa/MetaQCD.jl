function struct2dict(x::T) where {T}
    dict = Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))
    return dict
end

@kwdef mutable struct EnsembleParameters
    L::NTuple{4,Int64} = (4, 4, 4, 4)
    backend::String = "cpu"
    float_type::String = "float64"
    randomseed::Union{UInt64,Vector{UInt64}} = 0x0000000000000000
    NC::Int64 = 3
    numprocs_cart::NTuple{4,Int64} = (1, 1, 1, 1)
    halo_width::Int64 = 0
    numtherm::Int64 = 10
    numsteps::Int64 = 100
    inital::String = "cold"
    update_method::String = "HMC"
    hb_maxit::Int64 = 10^5
    numheatbath::Int64 = 4
    metro_epsilon::Float64 = 0.1
    metro_numhits::Int64 = 1
    metro_target_acc::Float64 = 0.5
    eo::Bool = true # XXX: Remove?
    or_algorithm::String = "subgroups"
    numorelax::Int64 = 0
    parity_update::Bool = false
end

@kwdef mutable struct DataParameters
    ensemble_dir::String = ""
    log_to_console::Bool = true
    verboselevel::Int64 = 1
    save_config_format::String = ""
    save_config_every::Int64 = 0
    save_checkpoint_format::String = ""
    save_checkpoint_every::Int64 = 0
    load_config_fromfile::Bool = false
    load_config_format::String = ""
    load_config_path::String = ""
    load_checkpoint_fromfile::Bool = false
    load_checkpoint_format::String = "jld2"
    load_checkpoint_path::String = ""
    overwrite::Bool = false
end

@kwdef mutable struct GaugeActionParameters
    beta::Float64 = 5.7
    gauge_action::String = "wilson"
end

@kwdef mutable struct FermionActionParameters
    fermion_action::String = "none"
    boundary_condition::String = "antiperiodic"
    eo_precon::Bool = false
    wilson_r::Float64 = 1.0
    wilson_csw::Float64 = 0.0
    fermions::Vector{Dict} = Dict[]
end

@kwdef mutable struct FermionParameters
    Nf::Union{Int,Vector{Int}} = 0
    mass::Union{Float64,Vector{Float64}} = 0.0
    cg_tol_action::Float64 = 1e-12
    cg_tol_md::Float64 = 1e-14
    cg_maxiters_action::Int64 = 1000
    cg_maxiters_md::Int64 = 1000
    rhmc_spectral_bound::NTuple{2,Float64} = (0.0, 64.0)
    rhmc_recalc_spectral_bound::Bool = false
    rhmc_order_action::Int64 = 15
    rhmc_order_md::Int64 = 10
    rhmc_prec_action::Int64 = 42
    rhmc_prec_md::Int64 = 42
end

@kwdef mutable struct BiasParameters
    rhostout_for_cv::Float64 = 0.12
    kinds_of_weights::Vector{String} = ["tiwari"]
    starting_Q::Union{Nothing,Vector{Int64}} = nothing
    usebiases::Vector{String} = [""]
    biases::Vector{Dict} = Dict[]
    # tempering specific
    tempering_enabled::Bool = false
    numinstances::Int64 = 1
    swap_every::Int64 = 1
    non_metadynamics_updates::Int64 = 1
    measure_on_all::Bool = false
end

@kwdef mutable struct HMCParameters
    hmc_trajectory::Float64 = 1
    hmc_steps::Int64 = 10
    hmc_friction::Float64 = 0.0
    hmc_integrator::String = "Leapfrog"
    hmc_rafriction::Float64 = 1.0
    hmc_numsmear_gauge::Int64 = 0
    hmc_numsmear_fermion::Int64 = 0
    hmc_rhostout_gauge::Float64 = 0.0
    hmc_rhostout_fermion::Float64 = 0.0
    hmc_logging::Bool = true
    levels::Vector{Dict} = Dict[]
end

@kwdef mutable struct GradientFlowParameters
    flow_integrator::String = "none"
    flow_num::Int64 = 0
    flow_tf::Float64 = 0.0
    flow_steps::Int64 = 0
    flow_measure_every::Union{Int64,Vector{Int64}} = 1
end

@kwdef mutable struct MeasurementParameters
    measurements::Vector{Dict} = Dict[]
    measurements_with_flow::Vector{Dict} = Dict[]
end
