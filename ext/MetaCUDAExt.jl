module MetaCUDAExt

using CUDA
using CUDA: CUDABackend, CuArray
import MetaQCD.Fields

function __init__()
    Fields.BACKENDS["cuda"] = CUDABackend
    return nothing
end

Fields.array_type(::Type{CUDABackend}) = CuArray

end
