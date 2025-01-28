# Creating a Parameter File

In order to run a simulation, a parameter file in the .toml format has to given as an input.
The parameters in MetaQCD.jl are divided into the categories: ["Physical Settings"](#physical-settings),
["Dynamical Fermion Settings"](#dynamical-fermion-settings), ["HMC Settings"](#hmc-settings),
["Gradient Flow Settings"](#gradient-flow-settings), ["Measurement Settings"](#measurement-settings),
["System Settings"](#system-settings) and ["Bias Settings"](#bias-settings). In a parameter file,
these categories should be used as keys for each class of parameters, e.g.:
```
["Physical Settings"]
L = [12, 12, 12, 12]
beta = 6.0
...
```

The default values for all of these are listed [here](./parameters.md#full-parameter-list-(=-default):).

## Physical Settings
- `L`: Lattice volume as a vector of integers (e.g., `L = [4, 4, 4, 4]`)
- `beta`: Gauge action beta parameter / coupling as a float (e.g., `beta = 6.0`)
- `gauge_action`: Type of gauge action as a string (e.g., `gauge_action = "wilson"`)
> Supported gauge actions: `"wilson"`, `"symanzik_tree"` (Lüscher-Weisz), `"iwasaki"`, `"dbw2"`

- `numtherm`: Number of thermalization updates as an integer (e.g., `numtherm = 100`)
- `numsteps`: Number of production updates as an integer (e.g., `numsteps = 100`)
- `initial`: Initial condition of the gauge field(s) as a string (e.g., `initial = "cold"`)
> Supported inital conditions: `"cold"` (identity), `"hot"` (random)

- `parity_update`: Whether or not there should be a 50% probability of hitting the gauge fields with a topological parity-flip as a boolean (e.g., `parity_update = true`)
- `update_method`: Update method as a string (e.g., `update_method = "hmc"`)
> Supported update methods: `"hmc"`, `"heatbath"`, `"metropolis"`

When using heatbath or metropolis, there are certain parameters specific to those methods
that can be set:
- `numheatbath`: Number of heatbath sweeps per update step as an integer (e.g, `numheatbath = 1`)
- `metro_numhits`: Number of metropolis hits per local update step as an integer (e.g, `metro_numhits = 10`)
- `metro_epsilon`: Metropolis starting proposal interval width (uniform) as a float (e.g, `metro_epsilon = 0.1`)
- `metro_target_acc`: Metropolis target acceptance probability as a float between 0 and 1 (e.g, `metro_target_acc = 0.5`)
- `or_algorithm`: Overrelaxation algorithm as a string (e.g, `or_algorithm = "subgroups"`)
> Supported overrelaxation algorithms: `"subgroups"` (Cabibbo-Marinari embedding), `"kenney-laub"`

## Dynamical Fermion Settings
- `fermion_action`: Fermion action definition as a string (e.g, `fermion_action = "staggered"`)
> Supported fermion actions: `"staggered"`, `"wilson"`, `"staggered_h1234"` (Hoelbling-type flavored mass term M12M34), `"staggered_h1342"` (Hoelbling-type flavored mass term M13M42)
- `eo_precon`: Whether to use an even-odd preconditioned operator as a string (e.g, `eo_precon = true`)
- `Nf`: Number of flavors per mass value as a vector of integers (e.g, `Nf = [2, 1, 1]` for 2+1+1 flavors)
- `mass`: Masses in terms of lattice units as a vector of floats (e.g, `mass = [0.001, 0.028, 0.1]`)
> Make sure `Nf` and `mass` have the same length!
- `csw`: Wilson-Clover improvement factor as a flow (e.g, `csw = 1.0`)
- `boundary_condition`: Boundary condition in time-direction as a string (e.g, `boundary_condition = "antiperiodic"`)

Parameters for the solver:
- `cg_tol_action`: CG tolerance for the calculation of the fermion determinant as a float (e.g, `cg_tol_action = 1e-12`)
- `cg_tol_md`: CG tolerance for the calculation of the fermion force as a float (e.g, `cg_tol_md = 1e-10`)
- `cg_maxiters_action`: Maximum amount of CG iterations for the calculation of the fermion determinant as an integer (e.g, `cg_maxiters_action = 1000`)
- `cg_maxiters_md`: Maximum amount of CG iterations for the calculation of the fermion force as an integer (e.g, `cg_maxiters_md = 1000`)

Parameters in case RHMC is used:
- `rhmc_spectral_bound`: Spectral bound for the rational approximation as a vector of floats (e.g, `rhmc_spectral_bound = [0.0, 64.0]`)
- `rhmc_order_action`: Order of the rational approximation of the fermion determinant as an integer (e.g, `rhmc_order_action = 15`)
- `rhmc_order_md`: Order of the rational approximation of the fermion force as an integer (e.g, `rhmc_order_md = 10`)
- `rhmc_prec_action`: Precision of the rational approximation of the fermion determinant as an integer (e.g, `rhmc_prec_action = 64`)
- `rhmc_prec_md`: Precision of the rational approximation of the fermion force as an integer (e.g, `rhmc_prec_md = 42`)

## HMC Settings
- `hmc_integrator`: Type of integrator to be used in HMC as a string (e.g, `hmc_integrator = "omf4"`)
> Supported HMC integrators: `"leapfrog"`, `"omf2"`, `"omf2slow"`, `"omf4"`, `"omf4slow"`
- `hmc_trajectory`: Trajectory length to be used in HMC as a float (e.g, `hmc_trajectory = 1`)
- `hmc_steps`: Number of integration steps per trajectory to be used in HMC as an integer (e.g, `hmc_steps = 10`)
- `hmc_rafriction`: Friction parameter for generalized HMC as a float (e.g, `hmc_steps = 10`)
- `hmc_numsmear_gauge`: Number of smearing steps for the gauge action in the HMC as an integer (e.g, `hmc_numsmear_gauge = 3`)
- `hmc_numsmear_fermion`: Number of smearing steps for the fermion action in the HMC as an integer (e.g, `hmc_numsmear_fermion = 4`)
- `hmc_rhostout_gauge`: Smearing step size for the gauge action in the HMC as a float (e.g, `hmc_rhostout_gauge = 0.12`)
- `hmc_rhostout_fermion`: Smearing step size for the fermion action in the HMC as a float (e.g, `hmc_rhostout_fermion = 0.125`)
- `hmc_logging`: Whether or not HMC data like accept-reject info or forces should be logged during the trajectory as boolean (e.g, `hmc_logging = true`)

## Gradient Flow Settings
- `flow_integrator`: Type of gradient flow integrator as a string (e.g, `flow_integrator = "rk3"`)
> Supported gradient flow integrators: `"euler"`, `"rk2"`, `"rk3"`, `"rk3w7"`, `"cooling"` (uses cooling smearing instead)
- `flow_num`: Number of flow trajectories to be peformed as a string (e.g, `flow_num = 100`)
- `flow_tf`: Step width per trajectory in gradient flow as a float (e.g, `flow_tf = 0.01`)
- `flow_steps`: Number of steps per gradient flow trajectory as an integer (e.g, `flow_steps = 10`)
- `flow_measure_every`: Specifies on after how many flow trajectories (or multiples of `flow_tf`) measurements should be taken.
If this is an integer, let's say 3, then observables are measured at every multiple of `3tf`.
If this is a vector of integers, then observables are measured at all the multiples of `tf` listed in the vector. (e.g, `flow_measure_every = [1, 2, 5, 10]`)
> Flowed configs CANNOT be written to file, only measurements on them are!

## Measurement Settings
The way measurements/observables are specified is quite different from the other parameters.

Firstly, there are two types of measurements: `measurements` and `measurements_with_flow`.
As the name suggests, the latter are measurements done on flowed configs, where the flow
parameters are given in the "Gradient Flow Settings". Again, measurements pertaining to one
of the two categories must be keyed by them with the subkey being the observale, e.g:
```
["Measurment Settings".measurements.Plaquette]
measure_every = 1
```
> Supported observables: `Plaquette`, `Polaykov_loop`, `Topological_charge`, `Energy_density`, `Gauge_action`, `Wilson_loop`, `Pion_correlator`

Each observable has the attribute `measure_every`, which specifies the interval between
measurements of that observable, meaning that all observables can have their own intervals.

Other observables, such as the `Topological_charge` or `Energy_density` have the additional attribute
`type`, which is a vector of strings, that contains the field strength tensor definition to be used.
> Supported field strength tensor definitions supported: `"plaquette"`, `"clover"`, "`improved`"

The observable `Gauge_action` also has the attribute `type`, again a vector of strings, that
contains the gauge action definitions to be measured. The supported gauge actions here, are the same
as in [Physical Settings](#physical-settings).

`Wilson_loop` is another special observable in that it has the integer-valued attributes `Rmax` and `Tmax`,
which specify the maximum width and length of the wilson loops to be measured.

The last special observable is the `Pion_correlator`, which has the attributes:
- `dirac_type`: 
- `eo_precon`: Whether to use an even-odd preconditioned operator as a string (e.g, `eo_precon = true`)
- `mass`: Masses in terms of lattice units as a float (e.g, `mass = 0.001`)
- `csw`: Wilson-Clover improvement factor as a flow (e.g, `csw = 1.0`)
- `boundary_condition`: Boundary condition in time-direction as a string (e.g, `boundary_condition = "antiperiodic"`)
- `cg_tol`: CG tolerance for the inversion of the operator as a float (e.g, `cg_tol = 1e-12`)
- `cg_maxiters`: Maximum amount of CG iterations for the inversion of the operator as an integer (e.g, `cg_maxiters = 1000`)

## System Settings
- `float_type`: The precision to be used for all gauge and spinor fields (global sums are always done in double precision) as a string (e.g, `float_type = "float32"`)
- `ensemble_dir`: Path to the ensemble generated by the simulation, that contains log files, measurement files, saved configs and biaspotentials as a string (e.g, `ensemble_dir = "/path/to/ensemble"`)
- `log_to_console`: Whether logs should be printed to console (they are always printed to file anyway) as a boolean (e.g, `print_to_console = false`)
- `verboselevel`: Level of verbosity of the logging. The higher the level, the more information is logged, as an integer (e.g, `verboselevel = 2`)
> Supported levels: 1 to 4 (To see what is left out as level goes lower, see code or try it out)
- `save_config_format`: The format configs should be written in as a string (e.g, `save_config_format = "bridge"`)
> Supported config formats: `"bridge"` (.txt file), `"jld"`, `"bmw"`
- `save_config_every`: Number of update iterations between writes of configs as an integer (e.g, `save_config_every = 10`)
- `save_checkpoint_format`: The format checkpoints should be stored in as a string (e.g, `save_checkpoint_format = "jld"`)
> Supported checkpoint formats: `"jld"`
- `save_checkpoint_every`: Number of update iterations between writes of checkpoints as an integer (e.g, `save_checkpoint_every = 10`)
- `load_config_fromfile`: Whether a config should be loaded from file to start from as a boolean (e.g, `load_config_fromfile = false`)
- `load_config_path`: Path to the file containing the config to be loaded as a string (e.g, `load_config_path = /path/to/config`)
- `load_config_format`: The format that the config specified in `load_config_path` is stored in as a string (e.g, `load_config_format = "bmw"`)
- `load_checkpoint_fromfile`: Whether a checkpoint should be loaded from file to start from as a boolean (e.g, `load_checkpoint_fromfile = false`)
- `load_checkpoint_path`: Path to the file containing the checkpoint to be loaded as a string (e.g, `load_checkpoint_path = /path/to/checkpoint`)
- `load_checkpoint_format`: The format that the checkpoint specified in `load_checkpoint_path` is stored in as a string (e.g, `load_checkpoint_format = "jld"`)
- `randomseed`: Random seed of the simulation as an unsigned integer (e.g, `randomseed = 0x0000000000000000`)
> If `randomseed == 0` then the seed is randomly set! (This is the default behavior)
- `overwrite`: Whether the program should ignore overwrites or error when a file would be overwritten as a boolean (e.g., `overwrite = true`)

## Bias Settings (Only when using biased enhanced sampling)
Biases are specified similarly to measurements. Note, that each bias/CV has to have the same smearing
step size, meaning that only the number of smearing steps can be varied per bias.
The general bias parameters are:
- `rhostout_for_cv`: Smearing step size for the cv as a float (e.g, `rhostout_for_cv = 0.12`)
- `numinstances`: Number of instances in PT-MetaD or multiple walkers as an integer (e.g, `numinstances = 2`)
- `starting_Q`: Topological sector each walker should start from when using multiple walkers as a vector of integers (e.g, `starting_Q = [-1, 0, 1]`)
> Length of `starting_Q` has to be equal to the number of walkers/instances!
- `tempering_enabled`: Whether PT-MetaD should be enabled as a boolean (e.g, `tempering_enabled = false`)
- `swap_every`: Number of update steps between swap proposals in PT-MetaD (e.g, `swap_every = 1`)
- `non_metadynamics_updates`: Number of update iterations the measurement stream should perform for every iteration the biased stream performs as an integer (e.g, `non_metadynamics_updates = 1`)
- `measure_on_all`: Whether all instances should measure observables as a boolean (always true for multiple walkers) (e.g, `measure_on_all = false`)

The individual biases/CVs themselves are specified like the observables in `measurements`,
albeit with the subkey `biases` followed by any ID of your choice:
```
["Bias Settings"]
rhostout_for_cv = 0.12
...
["Bias Settings".biases.1]
kind_of_bias = "metad"
...
["Bias Settings".biases.2]
kind_of_bias = "opes"
...
["Bias Settings".biases.2]
kind_of_bias = "opesmt"
...
```
Parameters that all bias types have, are:
- `kind_of_bias`: Type of bias as a string (e.g, `kind_of_bias = "metad"`)
> Supported bias types: `"metad"`, `"opes"`, `"opesmt"` (Multithermal OPES), `"parametric"`
- `kind_of_cv`: Collective variable to be used as a string (e.g, `kind_of_cv = "topcharge_clover"`)
> Supported cv types: `"topcharge_plaquette"`, `"topcharge_clover"`, `"multithermal"` (== gauge action, is default of `"opesmt"`)
- `numsmears_for_cv`: Number of smearing steps for the cv as an integer (e.g, `numsmears_for_cv = 4`)
- `static`: Whether the bias is static as a vector of booleans (one entry for each instance in PT-MetaD) (e.g, `static = [false]`)
- `symmetric`: Whether the bias is symmetric as a boolean (e.g, `symmetric = true`)
- `stride`: Number of production updates between bias updates a an integer (e.g, `stride = 10`)
- `cvlims`: Limits on the cv as an ordered vector of floats (e.g, `cvlims = [-3, 3]`)
- `penalty_weight`: Penalty weight when stepping outside of `cvlims` as a float (e.g, `penalty_weight = 100.0`)
- `biasfactor`: Biasfactor for a well-tempered bias as a float (e.g, `biasfactor = inf`)
- `kinds_of_weights`: Types of weights to be calculated for each config as a vector of strings (e.g, `kinds_of_weights = ["branduardi", "tiwari"]`)
> Supported weight types: `"tiwari"`, `"balanced_exp"`, `"branduardi"` (see: doi:10.1021/acs.jctc.9b00867)
- `usebiases`: Filenames of pre-build biases to be used as a vector of strings (one entry for each instance in PT-MetaD) (e.g, `usebiases = ["path_to_file"]`)
- `write_bias_every`: Interval between writes of bias to file as an integer (e.g, `write_bias_every = 10`)

Parameters in case a Metadynamics (`"metad"`) bias is used:
- `bin_width`: Bin width of bias as a float (e.g, `bin_width = 0.01`)
- `meta_weight`: Weight of Gaussians as a float (e.g, `meta_weight = 0.01`)

Parameters in case an OPES (`"opes"`) bias is used:
- `explore`: Whether OPES-Explore should be used as a boolean (e.g, `explore = false`)
- `barrier`: Approximate height of intersector barriers (should be slightly bigger than the actual height) as a float (e.g, `barrier = 10.0`)
- `sigma0`: Starting kernel width as a float (e.g, `sigma0 = 0.1`)
- `sigma_min`: Minimum kernel width as a float (e.g, `sigma_min = 1e-6`)
- `fixed_sigma`: Whether kernel width should be constant as a boolean (e.g, `fixed_sigma = false`)
- `no_Z`: Whether normalization Z should not be adaptive as a boolean (e.g, `no_Z = false`)

Parameters in case a OPES-Multithermal (`"opesmt"`) bias is used:
- `beta_min_max`: Minimum and Maximum of beta range as a vector of floats (e.g, `beta_min_max = [6.0, 6.5]`)
- `beta_num`: Number of betas the interval specified by `beta_min_max` should be divded by as an integer (e.g, `beta_num = 10`)
