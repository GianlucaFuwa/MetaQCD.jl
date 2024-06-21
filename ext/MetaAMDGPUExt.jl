module MetaAMDGPUExt

using AMDGPU: ROCBackend, ROCArray
import MetaQCD.Fields

function __init__()
    Fields.BACKEND["rocm"] = ROCBackend
    Fields.BACKEND["roc"] = ROCBackend
    Fields.BACKEND["amd"] = ROCBackend
    Fields.BACKEND["hip"] = ROCBackend
    return nothing
end

Fields.array_type(::Type{ROCBackend}) = ROCArray

end
