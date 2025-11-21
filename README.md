[![Global Docs](https://img.shields.io/badge/docs-MetaQCD-blue.svg)](https://gianlucafuwa.github.io/MetaQCD.jl/dev/)
<!-- [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) -->

# MetaQCD.jl

Inspired by the [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl/tree/master) package by Akio Tomiya et al.

For detailed information on how to use this package, see the [docs](https://gianlucafuwa.github.io/MetaQCD.jl/dev/).

## Features
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
- [x] RHMC to simulate odd number of flavours
- [x] Even-odd preconditioner for Wilson(-Clover)
- [x] Even-odd preconditioner for Staggered
- [ ] Mass-splitting preconditioner / Hasenbusch trick
- [x] Full support for GPU Backends (CUDA and ROCm for now)
- [x] Multi-node parallelism using MPI

## Installation
First make sure you have Julia version **1.9.4 or above** installed. You can use [juliaup](https://github.com/JuliaLang/juliaup) for that or just install the release from the [Julia website](https://julialang.org/downloads/oldreleases).
> The particular version you want is ultimately determined by the GPU you want to use.
Newer GPUs need more recent versions of Julia for compatibility.

Then:

1. Clone the latest release onto your machine.
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

If you want to use GPUs, make sure you not only have CUDA.jl or AMDGPU.jl installed, but also a fairly recent version of the CUDA Toolkit or ROCm.

## Quick Start
1. Set parameters using one of the templates in template folder (see docs for all the parameters and their options)
2. From shell, do:
```
julia --threads=auto metaqcd.jl parameters.toml
```

## Visualization
We include the ability to visualize your data (might not work as intended as of version 2.0.0) . For that, you just have to pass the directory where your ensemble lives:
```julia
] activate MetaAnalysis/
] instantiate
using MetaAnalysis
measurements = MetaMeasurements("/path/to/ensemble")
timeseries(measurements, :my_observable)
```

You can also create a holder of a bias potential and plot it. MetaQCD.jl creates the bias files with an extension that gives their type (.metad or .opes), but if you changed the extension you have to provide the bias type as a symbol under the kwarg `which`:
```julia
bias = MetaBias(myfile)
plot(bias)
```
