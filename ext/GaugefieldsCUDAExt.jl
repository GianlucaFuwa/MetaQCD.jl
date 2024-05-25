module GaugefieldsCUDAExt

using CUDA
using CUDA: CUDABackend, CuArray
using MetaQCD.Gaugefields

function __init__()
    Gaugefields.BACKEND["cuda"] = CUDABackend
    return nothing
end

Gaugefields.array_type(::Type{CUDABackend}) = CuArray

# Need to add this function to CUDA, because the base implementation is dynamic
CUDA.@device_override @noinline function Base.__throw_rational_argerror_typemin(
    ::Type{T}
) where {T}
    CUDA.@print_and_throw "invalid rational: denominator can't be typemin"
end

end
