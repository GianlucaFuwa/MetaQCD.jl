# Full Parameter list (= default):
```julia
Base.@kwdef mutable struct PhysicalParameters
    # gauge parameters
    L::NTuple{4,Int64} = (4, 4, 4, 4)
    beta::Float64 = 5.7
    NC::Int64 = 3
    gauge_action::String = "wilson"
    numtherm::Int64 = 10
    numsteps::Int64 = 100
    inital::String = "cold"
    update_method::Vector{String} = ["HMC"]
    hb_maxit::Int64 = 10^5
    numheatbath::Int64 = 4
    metro_epsilon::Float64 = 0.1
    metro_numhits::Int64 = 1
    metro_target_acc::Float64 = 0.5
    eo::Bool = true
    or_algorithm::String = "subgroups"
    numorelax::Int64 = 0
    parity_update::Bool = false
end

Base.@kwdef mutable struct DynamicalFermionParameters
    fermion_action::String = "none"
    Nf::Union{Int,Vector{Int}} = 0
    mass::Union{Float64,Vector{Float64}} = 0.0
    wilson_r::Float64 = 1.0
    wilson_csw::Float64 = 0.0
    anti_periodic::Bool = true
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
    eo_precon::Bool = false
end

Base.@kwdef mutable struct BiasParameters
    kind_of_bias::String = "none"
    kind_of_cv::String = "clover"
    numsmears_for_cv::Int64 = 4
    rhostout_for_cv::Float64 = 0.125
    is_static::Union{Bool,Vector{Bool}} = false
    symmetric::Bool = false
    stride::Int64 = 1
    cvlims::NTuple{2,Float64} = (-7, 7)
    biasfactor::Float64 = Inf
    kinds_of_weights::Vector{String} = ["tiwari"]
    usebiases::Vector{String} = [""]
    write_bias_every::Int64 = 1
    # metadynamics specific
    bin_width::Float64 = 1e-2
    meta_weight::Float64 = 1e-3
    penalty_weight::Float64 = 1000.0
    # opes specific
    barrier::Float64 = 0.0
    sigma0::Float64 = 0.1
    sigma_min::Float64 = 1e-6
    fixed_sigma::Bool = false
    no_Z::Bool = false
    opes_epsilon::Float64 = 0.0
    threshold::Float64 = 1.0
    cutoff::Float64 = 0.0
    # for parametric
    bias_Q::Float64 = 0.0
    bias_A::Float64 = 0.0
    bias_Z::Float64 = 0.0
    # tempering specific
    tempering_enabled::Bool = false
    numinstances::Int64 = 1
    swap_every::Int64 = 1
    non_metadynamics_updates::Int64 = 1
    measure_on_all::Bool = false
end

Base.@kwdef mutable struct HMCParameters
    hmc_trajectory::Float64 = 1
    hmc_steps::Int64 = 10
    hmc_friction::Float64 = 0.0
    hmc_integrator::String = "Leapfrog"
    hmc_numsmear_gauge::Int64 = 0
    hmc_numsmear_fermion::Int64 = 0
    hmc_rhostout_gauge::Float64 = 0.0
    hmc_rhostout_fermion::Float64 = 0.0
    hmc_logging::Bool = true
end

Base.@kwdef mutable struct GradientFlowParameters
    flow_integrator::String = "euler"
    flow_num::Int64 = 1
    flow_tf::Float64 = 0.1
    flow_steps::Int64 = 10
    flow_measure_every::Union{Int64,Vector{Int64}} = 1
end

Base.@kwdef mutable struct SystemParameters
    backend::String = "cpu"
    float_type::String = "float64"
    ensemble_dir::String = ""
    log_to_console::Bool = true
    verboselevel::Int64 = 1
    loadU_format::String = ""
    loadU_dir::String = ""
    loadU_fromfile::Bool = false
    loadU_filename::String = ""
    saveU_format::String = ""
    saveU_every::Int64 = 1
    randomseed::Union{UInt64,Vector{UInt64}} = 0x0000000000000000
    overwrite::Bool = false
end

Base.@kwdef mutable struct MeasurementParameters
    measurement_method::Vector{Dict} = Dict[]
end
