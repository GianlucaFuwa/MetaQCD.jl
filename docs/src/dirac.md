# Dirac Operators

Dirac operators are structs that hold a reference to the gauge background `U`, a temporary
fermion field in case one wants to use the doubly flavoured Hermitian variant, the bare mass
and the a Boolean indicating whether there are antiperiodic boundary conditions in the time
direction (yes, only periodic and antiperiodic BCs are supported so far).

To create a Dirac operators, use the constructors below. One thing to note is that dirac
operators can be constructed using any `Abstractfield` and so the gauge background is always
set to `nothing` on construction. In order to then add a gauge background you must use the
dirac operator as a functor on a `Gaugefield`, like `D_U = D_free(U)` (This does not
overwrite the `U` in `D_free` but creates a new dirac operator, that references the same
temporary fermion field as the parent).

```@docs
WilsonDiracOperator
```

```@docs
StaggeredDiracOperator
```

```@docs
StaggeredEOPreDiracOperator
```
