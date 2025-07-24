module MetaCUDAExt

using CUDA
using CUDA: @cuda, CUDABackend, CuArray, launch_configuration, synchronize
using CUDA: threadIdx, blockIdx, blockDim, reduce_block
import MetaQCD.Fields
import MetaQCD.Fields: _foreachindex_global!, _foreachindex_reduce_global!

function __init__()
    Fields.BACKENDS["cuda"] = CUDABackend
    return nothing
end

Fields.array_type(::Type{CUDABackend}) = CuArray
Fields.bzeros(::CUDABackend, args...) = CUDA.zeros(args...)
Fields.synchronize(::CUDABackend) = CUDA.synchronize()

function Fields.priority!(::CUDABackend, priority)
    CUDA.KernelAbstractions.priority!(CUDABackend(), priority)
    return nothing
end

function Fields.launch_foreachindex_global!(
    ::CUDABackend, f, captured, itr, threads, blocks
)
    @cuda threads=threads blocks=blocks _foreachindex_global!(f, captured, itr) 
    return nothing
end

function Fields.launch_foreachindex_reduce_global!(
    ::CUDABackend, out, op, f, captured, itr, threads, blocks
)
    out_vec = CUDA.zeros(typeof(out), blocks)
    wanted_items = nextpow(2, length(itr))
    reduce_items = wanted_items > threads ? prevpow(2, threads) : wanted_items
    bytes_out = sizeof(typeof(out)) * reduce_items
    @cuda threads=threads blocks=blocks shmem=bytes_out _foreachindex_reduce_global!(
        out_vec, out, op, f, captured, itr
    ) 
    return out_vec
end

@inline Fields.threadidx() = threadIdx().x
@inline Fields.groupidx() = blockIdx().x
@inline Fields.groupdim() = blockDim().x
@inline Fields.groupreduce(op, val, neutral) = reduce_block(op, val, neutral)

function Fields.simple_tune(itr, kernel, ::Type{CUDABackend})
    config = launch_configuration(kernel)
    return min(length(itr), config.threads)
end

end
