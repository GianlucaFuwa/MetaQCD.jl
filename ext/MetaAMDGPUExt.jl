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
Fields.priority!(::ROCBackend, priority) = AMDGPU.priority!(priority)

function Fields.mpi_assign_device!(::ROCBackend, id)
    Fields.DEVICE_ID[] != -1 && return nothing
    (0 <= id < AMDGPU.HIP.ndevices()) || throw(ArgumentError("Device id $id out of bounds."))
    AMDGPU.device_id!(Int32(id) + 1)
    Fields.DEVICE_ID[] = id
    return nothing
end

function Fields.launch_foreachindex_global!(
    ::ROCBackend, f, captured, itr, groupsize, gridsize
)
    if Fields.TUNE_KERNELS == Val(true)
        f_str = "$(Symbol(f))_$(Fields.float_type(captured[1]))"
        if !haskey(Fields.KERNEL_CACHE, f_str)
            kernel = @roc launch=false _foreachindex_global!(f, captured, itr)
            config = launch_configuration(kernel; max_block_size=min(length(itr), groupsize))
            Fields.KERNEL_CACHE[f_str] = config.groupsize
            groupsize = config.groupsize
        else
            groupsize = Fields.KERNEL_CACHE[f_str]
        end

        gridsize = cld(length(itr), groupsize)
    end

    @roc groupsize=groupsize gridsize=gridsize _foreachindex_global!(f, captured, itr) 
    return nothing
end

function Fields.launch_foreachindex_reduce_global!(
    ::ROCBackend, out, op, f, captured, itr, groupsize, gridsize
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
            out_vec = AMDGPU.zeros(typeof(out), gridsize)
            kernel = @roc launch=false _foreachindex_reduce_global!(
                out_vec, out, op, f, captured, itr
            ) 
            kernel_config = launch_configuration(kernel; shmem=max_shmem, max_block_size)
            # determine the launch configuration
            groupsize = compute_items(kernel_config.groupsize)
            gridsize = cld(length(itr), groupsize)
            Fields.KERNEL_CACHE[f_str] = groupsize
        else
            groupsize = Fields.KERNEL_CACHE[f_str]
            gridsize = cld(length(itr), groupsize)
        end
    end
    # perform the actual reduction
    out_vec = AMDGPU.zeros(typeof(out), groupsize)
    reduce_shmem = compute_shmem(groupsize)
    @roc gridsize=gridsize groupsize=groupsize shmem=reduce_shmem _foreachindex_reduce_global!(
        out_vec, out, op, f, captured, itr
    ) 
    return reduce(op, out_vec)
end

@inline Fields.threadidx() = workitemIdx().x
@inline Fields.groupidx() = workgroupIdx().x
@inline Fields.groupdim() = workgroupDim().x
@inline Fields.groupreduce(op, val, neutral) = reduce_group(op, val, neutral)

end
