[![Global Docs](https://img.shields.io/badge/docs-MetaQCD-blue.svg)](https://gianlucafuwa.github.io/MetaQCD.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# MetaQCD.jl (WIP full QCD branch---any commit might break something, message me, if you want a "stable" version)

Inspired by the [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl/tree/master) package by Akio Tomiya et al.

## Features:
- [x] Simulations of 4D-SU(3) Yang-Mills (Pure Gauge) theory
- [x] Simulations of full lattice QCD with arbitrary number of flavours (Staggered, Wilson-Clover)
- [x] [Metadynamics](https://www.researchgate.net/publication/224908601_Metadynamics_A_method_to_simulate_rare_events_and_reconstruct_the_free_energy_in_biophysics_chemistry_and_material_science)
- [x] [PT-MetaD](https://arxiv.org/abs/2307.04742)
- [x] Several update algorithms (HMC, Metropolis, Heatbath, Overrelaxation)
- [x] Several symplectic integrators for HMC (Leapfrog, OMF2, OMF4)
- [x] Gradient flow with variable integrators (Euler, RK2, RK3, RK3W7)
- [x] Improved Gauge actions (Symanzik tree, Iwasaki, DBW2)
- [x] Improved Topological charge definitions (clover, rectangle clover-improved)
- [x] Wilson(-Clover) fermions
- [x] Staggered fermions
- [ ] Even-odd preconditioner for Wilson(-Clover)
- [x] Even-odd preconditioner for Staggered
- [ ] Mass-splitting preconditioner / Hasenbusch trick
- [x] RHMC to simulate odd number of flavours
- [ ] Support for CUDA and ROCm backends
- [ ] Multi-device simulations using MPI

## Installation:
First make sure you have Julia version 1.9.x (or 1.10.x once [#2424](https://github.com/JuliaGPU/CUDA.jl/issues/2424) is fixed) installed. You can use [juliaup](https://github.com/JuliaLang/juliaup) for that or just install the release from the [Julia website](https://julialang.org/downloads/).

The package is not in the general registry. So you will have to either
- Add the package to your Julia environment (**not recommended**) via:
```
julia> ] add https://github.com/GianlucaFuwa/MetaQCD.jl
```

or (**recommended**, if you want to make changes yourself)

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

If you want to use a GPU, make sure you not only have CUDA.jl (v4.4.2) or AMDGPU.jl installed, but also a fairly recent version of the CUDA Toolkit or ROCm.

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
To use another backend, just append its name to the command:
```
julia --threads=auto metaqcd_sim.jl parameters.toml cuda
```

## Build a Bias:
1. Set parameters using the "parameters_build.toml" example in template folder
2. From shell, do:
```
julia --threads=auto metaqcd_build.jl parameters.toml
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
4. Begin build with prepared parameter file "parameters.toml":
``` julia
build_bias("parameters.toml")
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
