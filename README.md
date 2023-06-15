# MetaQCD.jl

Inspired by the LatticeQCD.jl package by Akio Tomiya et al.

## Quick Start:
1. Set parameters using one of the templates in template folder
2. From shell, do:
```
julia --threads=auto run.jl parameters.toml
```

or

2. Start Julia:
```
julia --threads=auto
```
3. Import MetaQCD package (this may prompt you to install dependencies):
``` julia
using MetaQCD
```
4. Begin Simulation with prepared parameter file "parameters.toml":
``` julia
@time run_sim("parameters.toml")
```
