# Fermion Actions

Instead of explicitly creating the Dirac operator, one can also create the corresponding
fermion action, with similar syntax. Some extra parameters that can be specified are the
number of flavors, RHMC-related parameters such as the spectral bounds and approximation
order & precision and solver-related parameters such as the tolerance and the maximum
number of solver iterations

```@docs
WilsonFermionAction
```

```@docs
StaggeredFermionAction
```

```@docs
StaggeredEOPreFermionAction
```

```@docs
StaggeredHoelblingFermionAction
```
