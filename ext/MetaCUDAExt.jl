module MetaCUDAExt

using CUDA
using CUDA: CUDABackend, CuArray
using MetaQCD.Fields

function __init__()
    Fields.BACKEND["cuda"] = CUDABackend
    return nothing
end

Fields.array_type(::Type{CUDABackend}) = CuArray

end
