module MetaAMDGPUExt

using AMDGPU
using AMDGPU: @roc, ROCBackend, ROCArray, launch_configuration, synchronize
using AMDGPU: workitemIdx, workgroupIdx, workgroupDim, reduce_group
import MetaQCD.Fields
import MetaQCD.Fields: _foreachindex_global!, _foreachindex_reduce_global!

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

function Fields.launch_foreachindex_global!(
    ::ROCBackend, f, captured, itr, groupsize, gridsize
)
    @roc groupsize=groupsize gridsize=gridsize _foreachindex_global!(f, captured, itr) 
    synchronize()
    return nothing
end

function Fields.launch_foreachindex_reduce_global!(
    ::ROCBackend, out, op, f, captured, itr, groupsize, gridsize
)
    out_vec = AMDGPU.zeros(typeof(out), gridsize)
    wanted_items = nextpow(2, length(itr))
    reduce_items = wanted_items > groupsize ? prevpow(2, groupsize) : wanted_items
    bytes_out = sizeof(typeof(out)) * reduce_items
    @roc groupsize=groupsize gridsize=gridsize shmem=bytes_out _foreachindex_reduce_global!(
        out_vec, out, op, f, captured, itr
    ) 
    return out_vec
end

@inline Fields.threadidx() = workitemIdx().x
@inline Fields.groupidx() = workgroupIdx().x
@inline Fields.groupdim() = workgroupDim().x
@inline Fields.groupreduce(op, val, neutral) = reduce_group(op, val, neutral)

function Fields.simple_tune(itr, kernel, ::Type{ROCBackend})
    config = launch_configuration(kernel)
    return min(length(itr), config.groupsize)
end

end
