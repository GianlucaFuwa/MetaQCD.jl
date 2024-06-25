# Creating Fields on the lattice

You can create a 4-dimensional SU(3) gauge field by specifying the backend, floating point
precision, gauge action, dimensions and coupling parameter beta:
```julia
backend = CPU
prec = Float64
action = WilsonGaugeAction
Ns = Nt = 12
beta = 6.0
U = Gaugefield{backend,prec,action}(Ns, Ns, Ns, Nt, beta) # all links are set to 0
```
and set the initial conditions with `identity_gauges!(U)` (cold) or
`random_gauges!(U)` (hot). \\
Gaugefields and the other two fields conatining group/algebra valued elements are structs
that contain a main Array `U`, which is a 5-dimensional array of statically sized 3x3
complex matrices, i.e., `SMatrix` objects from `StaticArrays.jl` (where arrays are stored as
Tuples under the hood. The different backends are handled by `Kernelabstractions.jl` which
exports a `zeros` method such that the array can be created on all supported backends with
no extra work. \\
The fact that the elements are statically sized immutable arrays means that, for one, there
are no allocations when performing linear algebra operations with them and secondly that we
always just override the matrices in the arrays instead of mutating them. This
yields enormous benefits in terms of less headaches during development and lets us define
custom linear algebra routines for SMatrices and SVectors. \\

There are no plans to use more memory efficient storage schemes for SU(3) or su(3) elements
yet. \\

Fermion fields or spinors or whatever you want to call them are stored in 4-dimensional 
arrays of `n_color * n_dirac` complex valued `SVector`s. The reason for chosing 4 instead of
5 dimensions is that this enabled us to write routines that take care of all dirac
components at the same time, which is more efficient (I think). \\
When using even-odd preconditioned dirac operators, the fermion fields get wrapped in a
struct called `EvenOdd` such that we can overload all functions on that type. Our convention
is to define the fields on the even sites. While we haven't tested whether the following is
actually more performant, we map all even sites to the first half of the array to have
contiguous memory accesses. The function `eo_site` does exactly this mapping. \\
`Fermionfield`s are created in the same way as `Gaugefield`s bar the gauge action type
parameter. For `Fermionfield`s we have the `ones!` and `gaussian_pseudofermions!` methods
to init them.

```@docs
Gaugefield
```

```@docs
Fermionfield
```

```@docs
Colorfield
```

```@docs
Expfield
```
