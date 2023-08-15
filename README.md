[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# MetaQCD.jl

Inspired by the [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl/tree/master) package by Akio Tomiya et al.

## Features:
- Simulations of 4D-SU(3) Yang-Mills (Pure Gauge) theory with and without Metadynamics
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
