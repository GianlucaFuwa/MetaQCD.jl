# Implementation of Gauge and Spinor Fields

This section describes the core field types in MetaQCD and how to create and initialize them for lattice QCD calculations.

## `AbstractField` Type Architecture

### Core Field Types

MetaQCD implements three main matrix field container types:

#### `Gaugefield`
- **Purpose**: Standard gauge field for lattice QCD simulations
- **Contains**: Main array `U` plus gauge action information
- **Usage**: Standard gauge field operations and measurements

#### `Colorfield` 
- **Purpose**: Simplified gauge field without action metadata
- **Contains**: Main array `U` only (no gauge action information)
- **Usage**: Intermediate calculations where action type is irrelevant

#### `Expfield`
- **Purpose**: Extended field for advanced smearing algorithms
- **Contains**: 3×3 matrices plus additional "Q" matrices for Stout algorithm
- **Usage**: Stout smearing and recursive smearing operations

### Data Storage Structure

On CPUs all field types use a **5-dimensional array** structure:
- **Dimensions**: `[μ, x, y, z, t]` where `μ` (fastest/first index) indexes the 4 spacetime directions
- **Elements**: Statically sized 3×3 complex matrices or 3ND-element complex vectors (`SMatrix`/`SVector` from StaticArrays.jl)

On GPUs we found that performance improved when making `μ` the slowest index. Additionally
we enforce vectorized loads and stores by not storing the matrices and vectors as is
but using the `SIMD.jl` type `Vec` to either group 2 or 4 floating point numbers together.
This essentially mimics the C++ types float2, float4 etc.

> Example 1: In the 12 element representation of SU(3) we store the matrices as 3 `SVec{4,T}` where `T` is the floating point precision
> Example 2: We store staggered spinors as 3 `SVec{2,T}`

When a field is indexed into, the `getindex` function automatically reconstructs the `SMatrix` or `SVec` from these SIMD vectors.

### Performance Benefits

The use of `StaticArrays` provides several advantages:

- **Zero Allocations**: No memory allocation during linear algebra operations
- **Immutable Operations**: Matrices are always replaced rather than mutated
- **Optimized Storage**: Compile-time known sizes enable aggressive optimization

### Backend Support

Different computing backends (CPU, GPU) are handled through Julia extensions, as detailed in the parallelization section. This allows the same code to run efficiently on various hardware while only loading GPU-specific code when needed.

### Future Optimizations

More memory-efficient storage schemes for SU(3) and su(3) elements (8 floating point representations)
may be implemented in future versions to further reduce memory footprint.

## Spinor Fields

### Data Structure

Fermion fields (spinors) are stored as **4-dimensional arrays** containing `n_color × n_dirac` complex-valued `SVector`s.

**Design Choice**: Using 4 dimensions instead of 5 enables writing routines that process all Dirac components simultaneously, improving computational efficiency.

### Creating Spinor Fields

```julia
# Create spinor field (n_dirac replaces gauge action parameter)
ψ = Spinorfield{backend,prec,n_dirac}(Ns, Ns, Ns, Nt)
```

### Initialization Methods

- **Unit field**: `ones!(ψ)` - Sets all components to 1
- **Random field**: `gaussian_pseudofermions!(ψ)` - Generates Gaussian random pseudofermions

## Even-Odd Preconditioning

### `EvenOdd` Wrapper

For even-odd preconditioned Dirac operators, spinor fields are wrapped in an `EvenOdd` struct:

```julia
ψ_eo = EvenOdd(ψ)
```

This wrapper allows overloading all relevant functions to work with the preconditioned structure.

### Memory Layout Optimization

**Convention**: Fields are defined on even sites, with all even sites mapped to the first half of the array for **contiguous memory access**.

**Implementation**: The `map_to_half` function handles the mapping between full lattice indices and the compressed even-site storage.

### Benefits

- **Reduced Memory**: Only stores even sites explicitly
- **Cache Efficiency**: Contiguous memory layout improves cache performance  
- **Algorithmic Efficiency**: Enables optimized even-odd preconditioned algorithms

## API Documentation

The following types are available with full documentation:

```@docs
Gaugefield
```

```@docs
Colorfield
```

```@docs
Expfield
```

```@docs
Spinorfield
```
