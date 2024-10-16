# Updating a Gaugefield

To update a `Gaugefield` simply use the `update!` function that takes 2 positional
arguments and however many keyword arguments specific to the update algorithm. 
The first positional is the actual update algorithm
[`update_alg`](@ref "Suppoorted Update Algorithms") and the second is the `Gaugefield` `U`.
```julia
U = Gaugefield(...)
random_gauges!(U)

MAXIT = 100
numHB = 1
or_alg = MeatQCD.Updates.Subgroups
numOR = 4
update_alg = MeatQCD.Updates.Heatbath(U, MAXIT, numHB, or_alg, numOR)

update!(update_alg, U; ...) 
```

## Supported Update Algorithms

```@docs
HMC
```

```@docs
Metropolis
```

```@docs
Heatbath
```

```@docs
Overrelaxation
```
