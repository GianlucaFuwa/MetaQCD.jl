module GaugefieldsAMDGPUExt

using AMDGPU: ROCBackend, ROCArray
import MetaQCD.Gaugefields

function __init__()
    Gaugefields.BACKEND["roc"] = ROCBackend
    Gaugefields.BACKEND["amd"] = ROCBackend
    return nothing
end

Gaugefields.array_type(::Type{ROCBackend}) = ROCArray

end
