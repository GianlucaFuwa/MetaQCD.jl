# Parallelization

This document explains how to use parallelization features in MetaQCD, including both single-node and distributed computing capabilities.

## Overview

All parallelization is handled through two main functions:
- `parallelfor` - For general parallel operations
- `parallelfor_sum` - For parallel reductions

Both functions are located in [`src/fields/utils/parallel.jl`](../../src/fields/utils/parallel.jl).

## Core Functions

### `parallelfor`

The main parallelization function with the following signature:

```julia
function parallelfor(
    f,                    # Kernel function to parallelize
    itr,                  # Iterator over lattice indices
    ::Type{B},            # Backend type (CPU/GPU)
    ::Val{M},             # MPI distribution flag (compile-time)
    to_validate::Tuple,   # Fields requiring halo validation before execution
    invalidated::Tuple,   # Fields whose halos become invalid after execution
    captured::Tuple;      # Fields captured by kernel f
    block_size=min(256, length(itr)),  # GPU block size (optional)
    do_edges=Val(false),  # Whether the edges and corners need to be respected in the halo exchange
    stream=default_stream(B())  # Which GPU stream to launch the kernel on
) where {B,M}
```

### Parameters

- **`f`**: The kernel function that will be executed in parallel
- **`itr`**: The space of lattice indices to iterate over
- **`B`**: Backend type, automatically embedded in parsed fields at compile time
- **`M`**: Boolean flag (wrapped in `Val`) indicating MPI distribution status
- **`to_validate`**: Tuple of fields whose halos must be validated before kernel execution
- **`invalidated`**: Tuple of fields whose halos become invalid after kernel execution
- **`captured`**: Fields that the kernel function `f` will access

## Backend Support

### CPU (Multithreading)
- Uses [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl)'s `@batch` macro
- Automatically handles thread distribution

### GPU Support
- Implemented via Julia extensions (loaded only when GPU packages are available)
- Key functions: `launch_foreachindex_global!` and `launch_foreachindex_reduce_global!`
- See [CUDA example](../../ext/MetaCUDAExt.jl) for implementation details

### Halo Exchange Optimization

The system optimizes performance by tracking halo validation status:
- Only validates halos that are out of date
- Marks halos as invalid when they're modified
- Saves computational time by avoiding unnecessary validation

## Communication Hiding

### Configuration

Communication can be hidden behind computation using the `HIDE_COMMS` compile-time variable. Configure this in `LocalPreferences.toml`:

```toml
[MetaQCD]
MPI_HIDE_COMMUNICATION = false  # Set to true to enable communication hiding
```

### How It Works

When `HIDE_COMMS = true`, the system:
1. Splits kernels into two parts:
   - First: Processes indices independent of halos
   - Second: Processes remaining indices
2. Executes halo exchange asynchronously using Julia's `@spawn` and task mechanism on CPUs
and hardware queues (streams) on GPUs
3. Overlaps computation with communication for better performance

### Kernel Tuning

Enable automatic kernel tuning in `LocalPreferences.toml`:

```toml
[MetaQCD]
TUNE_KERNELS = true
```

The tuning system uses built-in occupancy checkers from CUDA and ROCm to optimize performance beyond the default block size of 256.

## Usage Examples

### Parallel Reduction Example

```julia
function plaquette_trace_sum(U::Gaugefield{B,T,M}) where {B,T,M}
    P = parallelfor_sum(eachindex(U), 0.0, B, Val(M), (U,), (), (U,)) do pₙ, site, (U,)
        for μ in 1:3
            for ν in (μ+1):4
                pₙ += real(tr(plaquette(U, μ, ν, site)))
            end
        end
        return pₙ  # Must return reduction variable
    end
    return distributed_reduce(P, +, U)  # Reduce across all MPI ranks
end
```

**Note**: For reductions with `parallelfor_sum`, include an initial value (`0.0` in this example) after the iterator.

### Parallel Field Copy Example

```julia
function Base.copy!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, (a, b)
        a[μsite] = b[μsite]
    end
    return nothing
end
```

### Understanding the `do` Syntax

Julia's `do` syntax creates an anonymous function as the first argument:
- The code inside the `do` block becomes the kernel function `f`
- Arguments after `do` (like `pₙ, site, (U,)`) are the function parameters
- For reductions, the reduction variable must be the return value

## MPI Distributed Computing

### Domain Decomposition

Fields are distributed across MPI processes using a 4D tuple `numprocs_cart` that specifies the number of processes per spatial dimension.

### Topology Management

All topology information is stored in a [`FieldTopology`](../../src/fields/distributed/topology.jl) object, including:
- Halo width specifications
- Global and local dimensions/volumes

### Halo Implementation

Halos are implemented using **ghost cells**, i.e., padding around the original array.
This complex halo structure is **completely abstracted away** from the user through:

- **Overloaded `getindex` and `setindex!`**: These methods on `AbstractField` types automatically determine which halo array needs to be accessed based on the requested index
- **Simplified Indexing**: Using [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl), each partition's fields maintain the correct **global indices**, including halos
- **Transparent Usage**: Users can index fields naturally without worrying about the underlying halo storage implementation

This abstraction allows developers to write code as if working with a single, continuous array while benefiting from the optimized separate halo storage underneath.

### Halo Exchange Strategy

The system uses an **extended face propagation scheme** for handling edges and corners in halo exchanges. This approach simplifies implementation compared to handling edges and corners separately, making the code more maintainable and less error-prone.
