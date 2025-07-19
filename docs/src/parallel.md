# Parallelization

All paralleization is handled by the functions `parallelfor` and `parallelfor_sum` in
in the file [src/fields/utils/parallel.jl](../../src/fields/utils/parallel.jl).

```julia
function parallelfor(
    f,
    itr,
    ::Type{B}, # backend
    ::Val{M}, # whether field is mpi-distributed
    to_validate::Tuple,
    invalidated::Tuple,
    captured::Tuple;
    block_size=min(256, length(itr))
) where {B,M}
    return parallelfor(
        f, itr, B, Val(M), HIDE_COMMS, to_validate, invalidated, captured; block_size
    )
end
```

The way these functions work is that they take in a function `f` as the first argument which
would be the kernel for the function to be parallelized and as a second argument the space
of lattice indices `itr` to iterate over. The two next arguments are the backend `B` of the
fields and a Boolean `M` wrapped in a `Val` (to make it compile time known) that specifies
whether the fields are MPI-distributed.
For the halo exchange this function also needs to know which fields' halos have to be
validated before execution of the kernel and which fields' halos become invalidated after
execution of the kernel. In this way we can save time by not validating the halo of a field
whose halo is already up to date.
The last argument is the fields that are captured by the kernel `f` and one can optionally
pass the block size for execution on GPUs.

One may notice the `HIDE_COMMS` variable. This variable is set at compile time, by including
it in a `LocalPreferences.toml` file in the packages base directory like this:

```
[MetaQCD]
MPI_HIDE_COMMUNICATION = false
```

If true, communication is hidden behind computation by splitting the kernel into two, where
the first iterates over all indices which are independent of halos and secondly over the
rest, while executing the halo exchange asynchronously using Julia's task mechanism
and `@spawn`.

When using `parallelfor_sum` for reductions, there is one extra argument after `itr` which
is the initial value of the reduction variable `init`.

Multithreading (when `B == CPU`) is handled via [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl)'s
`@batch` macro and GPU execution via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
to be able to target different backends with very low coding overhead.

## Example Usage
An example usage of this parallelization function is:
```julia
function plaquette_trace_sum(U::Gaugefield{B,T,M}) where {B,T,M}
    P = parallelfor_sum(eachindex(U), 0.0, B, Val(M), (U,), (), (U,)) do pₙ, site, U
        for μ in 1:3
            for ν in (μ+1):4
                pₙ += real(tr(plaquette(U, μ, ν, site)))
            end
        end
        pₙ # reduction variable has to be the return value of the kernel for reductions
    end

    return distributed_reduce(P, +, U) # reduce over all MPI ranks that participated in the calculation
end

function Base.copy!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, a, b
        a[μsite] = b[μsite]
    end

    return nothing
end
```

In Julia one can use the `do` syntax for functions that take another function as their first
argument. The code within the `do` block is therefor the function `f` mentioned above
with the arguments `pₙ, site, U`.

## MPI distributed computing
MPI distribution of fields is done by splitting the fields according to a 4D tuple
`numprocs_cart` which specifies the number of processes per dimension. All the needed
information regarding the topology is then stored in a [`FieldTopology`](../../src/fields/distributed/topology.jl)
object, such as the halo width, the global/local dimensions/volume and the bulk, halo and
border indices.

Edges and corners in the halo exchange are handled by using an extended face propagation
scheme which is drastically easier to implement than doing edges and corners separately.
