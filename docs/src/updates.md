# Updating a Gaugefield

To update a `Gaugefield` simply use the `update!` function that takes 2 arguments. 
The first is the actual update algorithm [`update_alg`](@ref "Suppoorted Update Algorithms") and the second is the `Gaugefield` `U`.
```julia
U = Gaugefield(...)
random_gauges!(U)

update!(update_alg, U) 
```

## Supported Update Algorithms

```@docs
Metropolis
```

```@docs
Heatbath
```

```@docs
Overrelaxation
```

```@docs
HMC
```