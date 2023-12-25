[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# MetaQCD.jl

Inspired by the [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl/tree/master) package by Akio Tomiya et al.

## Features:
- Simulations of 4D-SU(3) Yang-Mills (Pure Gauge) theory
- [Metadynamics](https://www.researchgate.net/publication/224908601_Metadynamics_A_method_to_simulate_rare_events_and_reconstruct_the_free_energy_in_biophysics_chemistry_and_material_science)
- [PT-MetaD](https://arxiv.org/abs/2307.04742)
- Several update algorithms (HMC, Metropolis, Heatbath, Overrelaxation)
- Several symplectic integrators for HMC (Leapfrog, OMF2, OMF4)
- Gradient flow with variable integrators (Euler, RK2, RK3, RK3W7)
- Improved Gauge actions (Symanzik tree, Iwasaki, DBW2)
- Improved Topological charge definitions (clover, rectangle clover-improved)

## Installation:
First make sure you have a Julia version at or above 1.9.0 installed. You can use [juliaup](https://github.com/JuliaLang/juliaup) for that or just install the release from the [Julia website](https://julialang.org/downloads/).

The package in not in the general registry. So you will have to:
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
run_sim("parameters.toml")
```

## Visualization:
We include the ability to visualize your data. For that, you have to activate and instantiate the visualization project:
```julia
pkg> activate ./visualize
pkg> instantiate
```

Now you can create a holder for all measurements in a directory and plot a time series of an observable, specifying its filename (without extenstion) as a symbol:
```julia
measurements = MetaMeasurements(mydir)
timeseries(measurements, :myobservable)
```

You can also create a holder of a bias potential and plot it. MetaQCD.jl creates the bias files with an extension that gives their type (.metad or .opes), but if you changed the extension you have to provide the bias type as a symbol under the kwarg `which`:
```julia
bias = MetaBias(myfile, which=:mytype)
biaspotential(bias)
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
    metro_numhits::Int64 = 1
    metro_target_acc::Float64 = 0.5
    eo::Bool = true
    or_algorithm::String = "subgroups"
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

Base.@kwdef mutable struct PrintSystemParameters
    log_dir::String = ""
    log_to_console::Bool = true
    verboselevel::Int64 = 1
    loadU_format::String = ""
    loadU_dir::String = ""
    loadU_fromfile::Bool = false
    loadU_filename::String = ""
    saveU_dir::String = ""
    saveU_format::String = ""
    saveU_every::Int64 = 1
    randomseed::Union{UInt64, Vector{UInt64}} = 0x0000000000000000
    measurement_dir::String = ""
    bias_dir::Union{String, Vector{String}} = ""
    overwrite::Bool = false
end

Base.@kwdef mutable struct PrintHMCParameters
    hmc_trajectory::Float64 = 1
    hmc_steps::Int64 = 10
    hmc_friction::Float64 = π/2
    hmc_integrator::String = "Leapfrog"
    hmc_numsmear::Int64 = 0
    hmc_rhostout::Float64 = 0.0
end

Base.@kwdef mutable struct PrintGradientFlowParameters
    flow_integrator::String = "euler"
    flow_num::Int64 = 1
    flow_tf::Float64 = 0.1
    flow_steps::Int64 = 10
    flow_measure_every::Union{Int64, Vector{Int64}} = 1
end

Base.@kwdef mutable struct PrintMeasurementParameters
    measurement_method::Vector{Dict} = Dict[]
end
```
