module MetaAMDGPUExt

using AMDGPU
using AMDGPU: @roc, ROCBackend, ROCArray, launch_configuration, synchronize
using AMDGPU: workitemIdx, workgroupIdx, workgroupDim, reduce_group
import MetaQCD.Fields
import MetaQCD.Fields: _foreachindex_global!, _foreachindex_reduce_global!
import MetaQCD.Utils: mpi_myrank

function __init__()
    Fields.BACKENDS["rocm"] = ROCBackend
    Fields.BACKENDS["roc"] = ROCBackend
    Fields.BACKENDS["amd"] = ROCBackend
    Fields.BACKENDS["hip"] = ROCBackend
    return nothing
end

Fields.array_type(::Type{ROCBackend}) = ROCArray
Fields.bzeros(::ROCBackend, args...) = AMDGPU.zeros(args...)
Fields.synchronize(::ROCBackend) = AMDGPU.synchronize()
Fields.synchronize(::ROCBackend, stream) = AMDGPU.synchronize(stream)
Fields.device_synchronize(::ROCBackend) = AMDGPU.device_synchronize()
Fields.default_stream(::ROCBackend) = AMDGPU.stream()
Fields.priority!(::ROCBackend, priority) = AMDGPU.priority!(priority)

const roc_readstreams = Vector{AMDGPU.HIPStream}(undef, 0)
const roc_sendstreams = Vector{AMDGPU.HIPStream}(undef, 0)
const roc_streams = Vector{AMDGPU.HIPStream}(undef, 0)
const roc_priostreams = Vector{AMDGPU.HIPStream}(undef, 0)

function Fields.allocate_commstreams!(::ROCBackend, fields)
    global roc_readstreams, roc_sendstreams

    if length(fields) > length(roc_readstreams) ÷ 8
        push!(roc_readstreams, [AMDGPU.HIPStream(:high) for _ in 1:8, _ in 1:(length(fields)-length(roc_readstreams)÷8)]...)
    end

    if length(fields) > length(roc_sendstreams) ÷ 8
        push!(roc_sendstreams, [AMDGPU.HIPStream(:high) for _ in 1:8, _ in 1:(length(fields)-length(roc_sendstreams)÷8)]...)
    end

    return nothing
end

Fields.get_readstream(::ROCBackend, dir, dim, id) = roc_readstreams[dir + 2(dim-1) + 8(id-1)]
Fields.get_sendstream(::ROCBackend, dir, dim, id) = roc_sendstreams[dir + 2(dim-1) + 8(id-1)]

function Fields.get_stream(::ROCBackend, id::Integer)
    while id > length(roc_streams)
        push!(roc_streams, AMDGPU.HIPStream(:low))
    end

    return roc_streams[id]
end

function Fields.get_priority_stream(::ROCBackend, id::Integer)
    while id > length(roc_priostreams)
        push!(roc_priostreams, AMDGPU.HIPStream(:high))
    end

    return roc_priostreams[id]
end

function Fields.mpi_assign_device!(::ROCBackend, _id)
    id = mod(_id, AMDGPU.HIP.ndevices())
    Fields.FORCE_SINGLE_GPU == Val(true) && (id = 0)
    Fields.DEVICE_ID[] != -1 && return nothing
    (0 <= id < AMDGPU.HIP.ndevices()) || throw(ArgumentError("Device id $id out of bounds."))
    AMDGPU.device_id!(Int32(id + 1))
    Fields.DEVICE_ID[] = id
    dev = AMDGPU.device()
    Fields.MAX_SHMEM[] = AMDGPU.HIP.properties(dev).maxSharedMemoryPerMultiProcessor
    return nothing
end

function Fields.launch_foreachindex_global!(
    ::ROCBackend, f, captured, itr::Tuple, groupsize, stream=AMDGPU.stream()
)
    if Fields.TUNE_KERNELS == Val(true)
        f_str = "$(Symbol(f))_$(Fields.float_type(captured[1]))"
        if !haskey(Fields.KERNEL_CACHE, f_str) && (length(itr) == 1)
            # how many items do we want?
            wanted_items = nextpow(2, length(itr[1]))
            # how many items can we launch?
            max_block_size = min(1024, length(itr[1]))
            compute_items(max_items) = wanted_items > max_items ? prevpow(2, max_items) : wanted_items
            kernel = @roc launch=false _foreachindex_global!(f, captured, itr[1])
            config = launch_configuration(kernel; max_block_size)
            Fields.KERNEL_CACHE[f_str] = config.groupsize
            groupsize = config.groupsize
        else
            if haskey(Fields.KERNEL_CACHE, f_str)
                groupsize = Fields.KERNEL_CACHE[f_str]
            end
        end
    end

    # gridsize = if length(itr) == 1
    #     cld(length(itr[1]), groupsize)
    # else
    #     cld(sum(length.(itr)), groupsize)
    # end
    #
    # @roc groupsize=groupsize gridsize=gridsize stream=stream _foreachindex_global!(f, captured, itr...)

    gridsize = ntuple(i -> cld(length(itr[i]), groupsize), length(itr))

    for i in eachindex(itr)
        # _stream = length(itr) > 1 ? Fields.get_priority_stream(ROCBackend(), i+1) : stream 
        # _stream = stream
        @roc groupsize=groupsize gridsize=gridsize[i] stream=stream _foreachindex_global!(
            f, captured, itr[i]
        ) 
    end

    return nothing
end

function Fields.launch_foreachindex_reduce_global!(
    ::ROCBackend, out, op, f, captured, itr::Tuple, groupsize, stream=AMDGPU.stream()
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
            out_vec = AMDGPU.zeros(typeof(out), 256)
            kernel = @roc launch=false _foreachindex_reduce_global!(
                out_vec, out, op, f, captured, itr[1]
            ) 
            config = launch_configuration(kernel; shmem=max_shmem, max_block_size)
            # determine the launch configuration
            groupsize = compute_items(config.groupsize)
            Fields.KERNEL_CACHE[f_str] = groupsize
        else
            if haskey(Fields.KERNEL_CACHE, f_str)
                groupsize = Fields.KERNEL_CACHE[f_str]
            end
        end
    end

    gridsize = ntuple(i -> cld(length(itr[i]), groupsize), length(itr))

    # perform the actual reduction
    out_vec = AMDGPU.fill(out, length(itr) * maximum(gridsize))
    reduce_shmem = compute_shmem(groupsize)

    for i in eachindex(itr)
        # _stream = length(itr) > 1 ? Fields.get_priority_stream(ROCBackend(), i+1) : stream 
        @roc gridsize=gridsize[i] groupsize=groupsize shmem=reduce_shmem stream=stream _foreachindex_reduce_global!(
            out_vec, out, op, f, captured, itr[i],
        ) 
        # _stream != AMDGPU.stream() && AMDGPU.synchronize(_stream)
    end

    return reduce(op, out_vec)
end

@inline Fields.threadidx() = workitemIdx()
@inline Fields.groupidx() = workgroupIdx()
@inline Fields.groupdim() = workgroupDim()
@inline Fields.griddim() = gridGroupDim()
@inline Fields.groupreduce(op, val, neutral) = reduce_group(op, val, neutral)

end
