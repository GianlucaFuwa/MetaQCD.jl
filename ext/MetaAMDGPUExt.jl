module MetaAMDGPUExt

using AMDGPU: ROCBackend, ROCArray, launch_configuration
import MetaQCD.Fields

function __init__()
    Fields.BACKENDS["rocm"] = ROCBackend
    Fields.BACKENDS["roc"] = ROCBackend
    Fields.BACKENDS["amd"] = ROCBackend
    Fields.BACKENDS["hip"] = ROCBackend
    return nothing
end

Fields.array_type(::Type{ROCBackend}) = ROCArray

function Fields.simple_tune(itr, kernel, ::Type{ROCBackend})
    config = launch_configuration(kernel)
    return min(length(itr), config.groupsize)
end

end
