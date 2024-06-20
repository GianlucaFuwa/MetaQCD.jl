[![Global Docs](https://img.shields.io/badge/docs-MetaQCD-blue.svg)](https://gianlucafuwa.github.io/MetaQCD.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# MetaQCD.jl

Inspired by the [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl/tree/master) package by Akio Tomiya et al.

## Features:
- Simulations of 4D-SU(3) Yang-Mills (Pure Gauge) theory
- Simulations of full lattice QCD with arbitrary number of flavours (Staggered, Wilson-Clover)
- [Metadynamics](https://www.researchgate.net/publication/224908601_Metadynamics_A_method_to_simulate_rare_events_and_reconstruct_the_free_energy_in_biophysics_chemistry_and_material_science)
- [PT-MetaD](https://arxiv.org/abs/2307.04742)
- Several update algorithms (HMC, Metropolis, Heatbath, Overrelaxation)
- Several symplectic integrators for HMC (Leapfrog, OMF2, OMF4)
- Gradient flow with variable integrators (Euler, RK2, RK3, RK3W7)
- Improved Gauge actions (Symanzik tree, Iwasaki, DBW2)
- Improved Topological charge definitions (clover, rectangle clover-improved)
- Wilson fermions with and without clover improvement
- Staggered fermions
- Even-odd preconditioner
- RHMC to simulate odd number of flavours
- Support for CUDA and ROCm backends

## Installation:
First make sure you have a Julia version 1.9.x or 1.10.x installed. You can use [juliaup](https://github.com/JuliaLang/juliaup) for that or just install the release from the [Julia website](https://julialang.org/downloads/).

The package is not in the general registry. So you will have to either
- Add the package to your Julia environment (**not recommended**) via:
```
julia> ] add https://github.com/GianlucaFuwa/MetaQCD.jl
```

or (**recommended**)

1. Clone this repository onto your machine.
2. Open Julia in the directory which you cloned the repo into, with the project specific environment. This can either be done by starting Julia with the command line argument "--project" or by activating the environment within an opened Julia instance via the package manager:
``` julia
using Pkg
Pkg.activate(".")
```
Or you can switch to package manager mode by typing "]" and then do
``` julia
pkg> activate .
```
3. Instantiate the project to install all the dependencies using the package manager:
``` julia
Pkg.instantiate()
```
or
``` julia
pkg> instantiate
```

If you want to use a GPU, make sure you not only have CUDA.jl or AMDGPU.jl installed, but also a fairly recent version of the CUDA Toolkit or ROCm.

## Quick Start:
1. Set parameters using one of the templates in template folder
2. From shell, do:
```
julia --threads=auto metaqcd_sim.jl parameters.toml
```

or

2. Start Julia (with project):
```
julia --threads=auto --project=/path/to/dir/containing/MetaQCD.jl
```
3. Import MetaQCD package:
``` julia
using MetaQCD
```
4. Begin Simulation with prepared parameter file "parameters.toml":
``` julia
run_sim("parameters.toml")
```

## Visualization:
We include the ability to visualize your data. For that, you just have to pass the directory where your ensemble lives:
```julia
pkg> measurements = MetaMeasurements("my_ensemble")
pkg> timeseries(measurements, :my_observable)
```

You can also create a holder of a bias potential and plot it. MetaQCD.jl creates the bias files with an extension that gives their type (.metad or .opes), but if you changed the extension you have to provide the bias type as a symbol under the kwarg `which`:
```julia
bias = MetaBias(myfile, which=:mytype)
biaspotential(bias)
```

## Full Parameter list (= default):
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
```
