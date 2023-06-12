# MetaQCD.jl

Inspired by the LatticeQCD.jl package by Akio Tomiya et al.

## Quick Start:
- Set parameters using one of the templates in template folder
  - From shell, do:
  ```
  julia --threads=auto run.jl paramters.toml
  ```
or
  - Start Julia:
  ```
  julia --threads=auto
  ```
  - Import MetaQCD package (this may prompt you to install dependencies):
  ``` julia
  using MetaQCD
  ```
  - Begin Simulation with prepared parameter file "parameters.toml":
  ``` julia
  @time run_sim("parameters.toml")
  ```
