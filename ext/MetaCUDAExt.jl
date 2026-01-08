"""
This module defines all the functions required to make the code work on CUDA-based GPUs
by overloading the relevant functions in the mofule `Fields` with direct calls to CUDA.jl API.
Extension modules such as this one are loaded one both MetaQCD.jl and the "extension trigger"
module is called. The extensions and their triggers can be found in the Project.toml file
under `[extensions]`
"""
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
Fields.synchronize(::CUDABackend, stream) = CUDA.synchronize(stream)
Fields.device_synchronize(::CUDABackend) = CUDA.device_synchronize()
Fields.default_stream(::CUDABackend) = CUDA.stream()

function Fields.priority!(::CUDABackend, priority)
    CUDA.KernelAbstractions.priority!(CUDABackend(), priority)
    return nothing
end

const cu_readstreams = Vector{CUDA.CuStream}(undef, 0)
const cu_sendstreams = Vector{CUDA.CuStream}(undef, 0)
const cu_streams = Vector{CUDA.CuStream}(undef, 0)
const cu_priostreams = Vector{CUDA.CuStream}(undef, 0)

function Fields.allocate_commstreams!(::CUDABackend, fields)
    global cu_readstreams, cu_sendstreams

    # INFO: create 2 streams per dimension (4) per field (in the end the GPU will probably not)
    # not have as many hardware streams as are created here but that is not a problem
    if length(fields) > length(cu_readstreams) ÷ 8
        push!(cu_readstreams, [CUDA.CuStream(; priority=:high) for _ in 1:8, _ in 1:(length(fields)-length(cu_readstreams)÷8)]...)
    end

    if length(fields) > length(cu_sendstreams) ÷ 8
        push!(cu_sendstreams, [CUDA.CuStream(; priority=:high) for _ in 1:8, _ in 1:(length(fields)-length(cu_sendstreams)÷8)]...)
    end

    return nothing
end

Fields.get_readstream(::CUDABackend, dir, dim, id) = cu_readstreams[dir + 2(dim-1) + 8(id-1)]
Fields.get_sendstream(::CUDABackend, dir, dim, id) = cu_sendstreams[dir + 2(dim-1) + 8(id-1)]

function Fields.get_stream(::CUDABackend, id::Integer)
    while id > length(cu_streams)
        push!(cu_streams, CUDA.CuStream(; priority=:low))
    end

    return cu_streams[id]
end

function Fields.get_priority_stream(::CUDABackend, id::Integer)
    while id > length(cu_priostreams)
        push!(cu_priostreams, CUDA.CuStream(; priority=:high))
    end

    return cu_priostreams[id]
end

function Fields.mpi_assign_device!(::CUDABackend, _id)
    id = mod(_id, CUDA.ndevices())
    Fields.FORCE_SINGLE_GPU == Val(true) && (id = 0)
    Fields.DEVICE_ID[] != -1 && return nothing
    (0 <= id < CUDA.ndevices()) || throw(ArgumentError("Device id $id out of bounds."))
    CUDA.device!(Int32(id))
    Fields.DEVICE_ID[] = id
    dev = CUDA.device()
    Fields.MAX_SHMEM[] = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return nothing
end

function Fields.launch_foreachindex_global!(
    ::CUDABackend, f, captured, itr::Tuple, threads, stream=CUDA.stream()
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
            config = launch_configuration(kernel.fun; max_threads)
            Fields.KERNEL_CACHE[f_str] = config.threads
            threads = compute_items(config.threads)
        else
            if haskey(Fields.KERNEL_CACHE, f_str)
                threads = Fields.KERNEL_CACHE[f_str]
            end
        end
    end

    blocks = ntuple(i -> cld(length(itr[i]), threads), length(itr))

    for i in eachindex(itr)
        @cuda threads=threads blocks=blocks[i] stream=stream _foreachindex_global!(
            f, captured, itr[i]
        ) 
    end

    return nothing
end

function Fields.launch_foreachindex_reduce_global!(
    ::CUDABackend, out, op, f, captured, itr::Tuple, threads, stream=CUDA.stream()
)
    length(itr) == 0 && return out
    compute_shmem(items) = items * sizeof(typeof(out))

    if Fields.TUNE_KERNELS == Val(true)
        f_str = "$(Symbol(f))_$(Fields.float_type(captured[1]))"
        if !haskey(Fields.KERNEL_CACHE, f_str) && (length(itr) == 1)
            # how many items do we want?
            wanted_items = nextpow(2, length(itr[1]))
            # how many items can we launch?
            max_threads = min(1024, length(itr[1]))
            compute_items(max_items) = wanted_items > max_items ? prevpow(2, max_items) : wanted_items
            max_shmem = max_threads |> compute_items |> compute_shmem
            out_vec = CUDA.zeros(typeof(out), 256)
            kernel = @cuda launch=false _foreachindex_reduce_global!(
                out_vec, out, op, f, captured, itr[1]
            ) 
            config = launch_configuration(kernel.fun; shmem=max_shmem, max_threads)
            # determine the launch configuration
            threads = compute_items(config.threads)
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
        @cuda threads=threads blocks=blocks[i] shmem=reduce_shmem stream=stream _foreachindex_reduce_global!(
            out_vec, out, op, f, captured, itr[i]
        ) 
    end

    return reduce(op, out_vec)
end

@inline Fields.threadidx() = threadIdx()
@inline Fields.groupidx() = blockIdx()
@inline Fields.groupdim() = blockDim()
@inline Fields.griddim() = gridDim()
# Since this is currently only used for sum reductions, we can assume associativity (in exact arithmetic) and set shuffle = True
@inline Fields.groupreduce(op, val, neutral) = reduce_block(op, val, neutral, #=shuffle=# Val(True))

end
