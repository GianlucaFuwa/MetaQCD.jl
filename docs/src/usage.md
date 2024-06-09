# Usage

If you just want to perform a simulation with some parameters, then

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

Logs, measurements and the lot are all written to files in the `ensembles` directory under
the specified directory name. If no directory name is provided, one is generated based on
time the simulation was started at.
