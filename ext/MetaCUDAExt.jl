module MetaCUDAExt

using CUDA
using CUDA: @cuda, CUDABackend, CuArray, launch_configuration, synchronize
using CUDA: threadIdx, blockIdx, blockDim, reduce_block
import MetaQCD.Fields
import MetaQCD.Fields: _foreachindex_global!, _foreachindex_reduce_global!
import MetaQCD.Utils: mpi_myrank

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

function Fields.mpi_assign_device!(::CUDABackend, id)
    # TODO: remove this variable and just do id % numdevices
    Fields.FORCE_SINGLE_GPU == Val(true) && (id = 0)
    Fields.DEVICE_ID[] != -1 && return nothing
    (0 <= id < CUDA.ndevices()) || throw(ArgumentError("Device id $id out of bounds."))
    CUDA.device!(Int32(id))
    Fields.DEVICE_ID[] = id
    dev = AMDGPU.device()
    Fields.MAX_SHMEM[] = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return nothing
end

function Fields.launch_foreachindex_global!(
    ::CUDABackend, f, captured, itr::Tuple, threads
)
    if Fields.TUNE_KERNELS == Val(true)
        f_str = "$(Symbol(f))_$(Fields.float_type(captured[1]))"
        if !haskey(Fields.KERNEL_CACHE, f_str) && (length(itr) == 1)
            # how many items do we want?
            wanted_items = nextpow(2, length(itr[1]))
            # how many items can we launch?
            max_threads = min(1024, length(itr[1]))
            compute_items(max_items) = wanted_items > max_items ? prevpow(2, max_items) : wanted_items
            kernel = @cuda launch=false _foreachindex_global!(f, captured, itr[1])
            config = launch_configuration(kernel; max_threads)
            Fields.KERNEL_CACHE[f_str] = config.threads
            threads = compute_items(config.groupsize)
        else
            if haskey(Fields.KERNEL_CACHE, f_str)
                threads = Fields.KERNEL_CACHE[f_str]
            end
        end
    end

    blocks = ntuple(i -> cld(length(itr[i]), threads), length(itr))

    for i in eachindex(itr)
        @cuda threads=threads blocks=blocks[i] _foreachindex_global!(
            f, captured, itr[i]
        ) 
    end

    return nothing
end

function Fields.launch_foreachindex_reduce_global!(
    ::CUDABackend, out, op, f, captured, itr::Tuple, threads, blocks
)
    length(itr) == 0 && return out
    compute_shmem(items) = items * sizeof(typeof(out))

    if Fields.TUNE_KERNELS == Val(true)
        f_str = "$(Symbol(f))_$(Fields.float_type(captured[1]))"
        if !haskey(Fields.KERNEL_CACHE, f_str) && (length(itr) == 1)
            # how many items do we want?
            wanted_items = nextpow(2, length(itr[1]))
            # how many items can we launch?
            max_block_size = min(1024, length(itr[1]))
            compute_items(max_items) = wanted_items > max_items ? prevpow(2, max_items) : wanted_items
            max_shmem = max_block_size |> compute_items |> compute_shmem
            out_vec = CUDA.zeros(typeof(out), 256)
            kernel = @cuda launch=false _foreachindex_reduce_global!(
                out_vec, out, op, f, captured, itr[1], UInt(8)
            ) 
            config = launch_configuration(kernel; shmem=max_shmem, max_block_size)
            # determine the launch configuration
            threads = compute_items(config.groupsize)
            Fields.KERNEL_CACHE[f_str] = threads
        else
            if haskey(Fields.KERNEL_CACHE, f_str)
                threads = Fields.KERNEL_CACHE[f_str]
            end
        end
    end

    blocks = ntuple(i -> cld(length(itr[i]), threads), length(itr))

    # perform the actual reduction
    out_vec = CUDA.fill(out, length(itr) * maximum(blocks))
    reduce_shmem = compute_shmem(threads)
    for i in eachindex(itr)
        @cuda blocks=_blocks[i] threads=threads shmem=reduce_shmem _foreachindex_reduce_global!(
            out_vec, out, op, f, captured, itr[i], UInt8(i)
        ) 
    end

    return reduce(op, out_vec)
end

@inline Fields.threadidx() = threadIdx()
@inline Fields.groupidx() = blockIdx()
@inline Fields.groupdim() = blockDim()
@inline Fields.griddim() = gridDim()
@inline Fields.groupreduce(op, val, neutral) = reduce_block(op, val, neutral)

end
