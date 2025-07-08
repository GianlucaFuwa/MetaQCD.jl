module MetaCUDAExt

using CUDA
using CUDA: CUDABackend, CuArray, launch_configuration
import MetaQCD.Fields

function __init__()
    Fields.BACKENDS["cuda"] = CUDABackend
    return nothing
end

Fields.array_type(::Type{CUDABackend}) = CuArray

function Fields.simple_tune(itr, kernel, ::Type{CUDABackend})
    config = launch_configuration(kernel)
    return min(length(itr), config.threads)
end

end
