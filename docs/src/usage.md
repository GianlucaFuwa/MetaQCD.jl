# Usage

If you just want to perform a simulation with some parameters, then

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

Logs, measurements and the lot are all written to files in the `ensembles` directory under
the specified directory name. If no directory name is provided, one is generated based on
time the simulation was started at.

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
