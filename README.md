[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# MetaQCD.jl

Inspired by the [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl/tree/master) package by Akio Tomiya et al.

## Features:
- Simulations of 4D-SU(3) Yang-Mills (Pure Gauge) theory
- [Metadynamics](https://www.researchgate.net/publication/224908601_Metadynamics_A_method_to_simulate_rare_events_and_reconstruct_the_free_energy_in_biophysics_chemistry_and_material_science)
- [PT-MetaD](https://arxiv.org/abs/2307.04742)
- Several update algorithms (HMC, Metropolis, Heatbath, Overrelaxation)
- Several symplectic integrators for HMC (Leapfrog, OMF2, OMF4) -> Case sensititve in parameter-files
- Gradient flow with variable integrators (Euler, RK2, RK3, RK3W7) -> Case sensititve in parameter-files
- Improved Gauge actions (Symanzik tree, Iwasaki, DBW2)
- Improved Topological charge definitions (clover, rectangle clover-improved)

## Quick Start:
1. Set parameters using one of the templates in template folder
2. From shell, do:
```
julia --threads=auto run.jl parameters.toml
```

or

2. Start Julia (with project):
```
julia --threads=auto --project
```
3. Import MetaQCD package (this may prompt you to install dependencies):
``` julia
using MetaQCD
```
4. Begin Simulation with prepared parameter file "parameters.toml":
``` julia
@time run_sim("parameters.toml")
```
## Full Parameter list (= default):
```julia
Base.@kwdef mutable struct PrintPhysicalParameters
    L::NTuple{4, Int64} = (4, 4, 4, 4)
    beta::Float64 = 5.7
    NC::Int64 = 3
    kind_of_gaction::String = "wilson"
    numtherm::Int64 = 10
    numsteps::Int64 = 100
    inital::String = "cold"
    update_method::Vector{String} = ["HMC"]
    hb_maxit::Int64 = 10^5
    numheatbath::Int64 = 4
    metro_epsilon::Float64 = 0.1
    metro_multi_hit::Int64 = 1
    metro_target_acc::Float64 = 0.5
    eo::Bool = false
    numorelax::Int64 = 0
    parity_update::Bool = false
end

Base.@kwdef mutable struct PrintBiasParameters
    kind_of_bias::String = "none"
    kind_of_cv::String = "clover"
    numsmears_for_cv::Int64 = 4
    rhostout_for_cv::Float64 = 0.125
    is_static::Union{Bool, Vector{Bool}} = false
    symmetric::Bool = false
    stride::Int64 = 1
    cvlims::NTuple{2, Float64} = (-7, 7)
    biasfactor::Float64 = Inf
    kinds_of_weights::Vector{String} = ["tiwari"]
    usebiases::Union{Nothing, String, Vector{Union{Nothing,String}}} = nothing
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
    # tempering specific
    tempering_enabled::Bool = false
    numinstances::Int64 = 1
    swap_every::Int64 = 1
    non_metadynamics_updates::Int64 = 1
    measure_on_all::Bool = false
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
    randomseed::Int64 = 0
    measurement_basedir::String = ""
    measurement_dir::String = ""
    bias_basedir::Union{Nothing, String, Vector{String}} = nothing
    bias_dir::Union{Nothing, String, Vector{Union{Nothing,String}}} = nothing
    overwrite::Bool = false
end

Base.@kwdef mutable struct PrintHMCParameters
    hmc_deltatau::Float64 = 0.1
    hmc_steps::Int64 = 10
    hmc_friction::Float64 = π/2
    hmc_integrator::String = "Leapfrog"
    hmc_numsmear::Int64 = 0
    hmc_rhostout::Float64 = 0.0
end

Base.@kwdef mutable struct PrintGradientFlowParameters
    hasgradientflow::Bool = false
    flow_integrator::String = "euler"
    flow_num::Int64 = 1
    flow_tf::Float64 = 0.1
    flow_steps::Int64 = 10
    flow_measure_every::Int64 = 1
end
```
