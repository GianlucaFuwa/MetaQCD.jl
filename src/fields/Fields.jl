module Fields

using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Polyester # Used for the @batch macro, which enables multi threading
using Random
using StaticArrays # Used for the SU3 matrices
using ..Utils # Contains utility functions, such as projections and the exponential map

import KernelAbstractions as KA # With this we can write generic GPU kernels for ROC and CUDA
import StrideArraysCore: object_and_preserve # This is used to convert the Abstractfield to a PtrArray in the @batch loop

# When CUDA.jl or AMDGPU.jl are loaded, their backends are appended to this Dict
const BACKEND = Dict{String,Type{<:Backend}}("cpu" => CPU)

# We are going to need these if we want to transfer a field from one backend to another
# For other backends, we overload this method in their respective extensions
array_type(::Type{CPU}) = Array

# Define an abstract field super type that is parametrized by the backend, the precision and
# the array type (Array, CuArray, ROCArray)
abstract type Abstractfield{B,T,A} end

include("gaugefields.jl")
include("liefields.jl")
include("fieldstrength.jl")
include("fermionfields.jl")
include("iterators.jl")
include("gpu_iterators.jl")
include("gpu_kernels/utils.jl")

include("field_operations.jl")
include("wilsonloops.jl")
include("actions.jl")
include("staples.jl")
include("clovers.jl")

include("gpu_kernels/field_operations.jl")
include("gpu_kernels/wilsonloops.jl")
include("gpu_kernels/actions.jl")
include("gpu_kernels/liefields.jl")
include("gpu_kernels/fieldstrength.jl")
include("gpu_kernels/fermionfields.jl")

end
