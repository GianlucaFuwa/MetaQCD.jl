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
    if Fields.TUNE_KERNELS == Val(true)
        f_str = "$(Symbol(f))_$(Fields.float_type(captured[1]))"
        if !haskey(Fields.KERNEL_CACHE, f_str)
            kernel = @cuda launch=false _foreachindex_global!(f, captured, itr)
            config = launch_configuration(kernel; max_threads=min(length(itr), threads))
            Fields.KERNEL_CACHE[f_str] = config.threads
            threads = config.groupsize
        else
            threads = Fields.KERNEL_CACHE[f_str]
        end

        blocks = cld(length(itr), threads)
    end

    @cuda threads=threads blocks=blocks _foreachindex_global!(f, captured, itr) 
    return nothing
end

function Fields.launch_foreachindex_reduce_global!(
    ::CUDABackend, out, op, f, captured, itr, threads, blocks
)
    length(itr) == 0 && return out
    compute_shmem(items) = items * sizeof(typeof(out))

    if Fields.TUNE_KERNELS == Val(true)
        f_str = "$(Symbol(f))_$(Fields.float_type(captured[1]))"
        if !haskey(Fields.KERNEL_CACHE, f_str)
            # how many items do we want?
            wanted_items = nextpow(2, length(itr))
            # how many items can we launch?
            max_block_size = 1024
            compute_items(max_items) = wanted_items > max_items ? prevpow(2, max_items) : wanted_items
            max_shmem = max_block_size |> compute_items |> compute_shmem
            out_vec = CUDA.zeros(typeof(out), threads)
            kernel = @cuda launch=false _foreachindex_reduce_global!(
                out_vec, out, op, f, captured, itr
            ) 
            kernel_config = launch_configuration(kernel; shmem=max_shmem, max_block_size)
            # determine the launch configuration
            threads = compute_items(kernel_config.groupsize)
            Fields.KERNEL_CACHE[f_str] = threads
        else
            threads = Fields.KERNEL_CACHE[f_str]
        end

        blocks = cld(length(itr), threads)
    end
    # perform the actual reduction
    out_vec = CUDA.zeros(typeof(out), threads)
    reduce_shmem = compute_shmem(threads)
    @cuda blocks=blocks threads=threads shmem=reduce_shmem _foreachindex_reduce_global!(
        out_vec, out, op, f, captured, itr
    ) 
    return reduce(op, out_vec)
end

@inline Fields.threadidx() = threadIdx().x
@inline Fields.groupidx() = blockIdx().x
@inline Fields.groupdim() = blockDim().x
@inline Fields.groupreduce(op, val, neutral) = reduce_block(op, val, neutral)

end
