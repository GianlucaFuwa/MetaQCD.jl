# MetaQCD.jl

Inspired by the [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl/tree/master) package by Akio Tomiya et al.

## Features
- Simulations of 4D-SU(3) Yang-Mills (Pure Gauge) theory
- [Metadynamics](https://www.researchgate.net/publication/224908601_Metadynamics_A_method_to_simulate_rare_events_and_reconstruct_the_free_energy_in_biophysics_chemistry_and_material_science)
- [PT-MetaD](https://arxiv.org/abs/2307.04742)
- Several update algorithms (HMC, Metropolis, Heatbath, Overrelaxation)
- Several symplectic integrators for HMC (Leapfrog, OMF2, OMF4)
- Gradient flow with variable integrators (Euler, RK2, RK3, RK3W7)
- Improved Gauge actions (Symanzik tree, Iwasaki, DBW2)
- Improved Topological charge definitions (clover, rectangle clover-improved)
- Support for CUDA and ROCm backends

## Installation
First make sure you have a Julia version at or above 1.9.0 installed. You can use [juliaup](https://github.com/JuliaLang/juliaup) for that or just install the release from the [Julia website](https://julialang.org/downloads/).

The package in not in the general registry. So you will have to either
- Add the package to your Julia environment via:
```
julia> ] add https://github.com/GianlucaFuwa/MetaQCD.jl
```

or

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
