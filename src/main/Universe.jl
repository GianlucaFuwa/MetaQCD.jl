module Universe

using AMDGPU: ROCBackend
using CUDA: CUDABackend
using Dates
using KernelAbstractions: CPU
using Unicode
using ..Utils
using ..Output

import ..Gaugefields: Gaugefield, Plaquette, Clover
import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
    SymanzikTadGaugeAction, SymanzikTreeGaugeAction, WilsonGaugeAction
import ..Gaugefields: initial_gauges
import ..BiasModule: Bias, get_cvtype_from_parameters, write_to_file
import ..Parameters: ParameterSet
import ..Smearing: NoSmearing, StoutSmearing

struct Univ{TG,TB}
    U::TG
    bias::TB
    numinstances::Int64
end

function Univ(p::ParameterSet; use_mpi=false, fp=true)
    @level1("┌ Setting Universe...")
    NX, NY, NZ, NT = p.L
    β = p.beta
    GA = get_gaugeaction_from_string(p.kind_of_gaction)
    backend = get_backend_from_string(p.backend)
    T = get_floatT_from_string(p.float_type)

    @level1("|  L: $(NX)x$(NY)x$(NZ)x$(NT)")
    @level1("|  GaugeAction: $(GA)")
    @level1("|  beta: $β")
    @level1("└\n")

    if p.kind_of_bias != "none"
        if p.tempering_enabled && use_mpi==false
            numinstances = p.numinstances
            @level1("|  Using 1 + $(numinstances-1) instances\n")
            U₁ = initial_gauges(p.initial, NX, NY, NZ, NT, β, GA=GA, backend=backend, T=T)
            bias₁ = Bias(p, U₁; instance=0)

            U = Vector{typeof(U₁)}(undef, numinstances)
            bias = Vector{typeof(bias₁)}(undef, numinstances)

            bias[1] = bias₁
            U[1] = U₁
            for i in 2:numinstances
                U[i] = initial_gauges(p.initial, NX, NY, NZ, NT, β, GA=GA, backend=backend, T=T)
                bias[i] = Bias(p, U[i]; instance=i-1)
            end
        else
            numinstances = 1
            U = initial_gauges(p.initial, NX, NY, NZ, NT, β, GA=GA, backend=backend, T=T)
            bias = Bias(p, U)
        end
    else
        @assert p.tempering_enabled == false "tempering can only be enabled with bias"
        numinstances = 1
        U = initial_gauges(p.initial, NX, NY, NZ, NT, β, GA=GA, backend=backend, T=T)
        bias = nothing
    end

    return Univ{typeof(U), typeof(bias)}(U, bias, numinstances)
end

function get_gaugeaction_from_string(gaction::String)
    kind_of_gaction = Unicode.normalize(gaction, casefold=true)

    if kind_of_gaction == "wilson"
        GA = WilsonGaugeAction
    elseif kind_of_gaction == "symanzik_tree"
        GA = SymanzikTreeGaugeAction
    # elseif kind_of_gaction == "symanzik_tadpole"
    #     GA = SymanzikTadGaugeAction
    elseif kind_of_gaction == "iwasaki"
        GA = IwasakiGaugeAction
    elseif kind_of_gaction == "dbw2"
        GA = DBW2GaugeAction
    else
        error("Gauge action \"$(gaction)\" not supported")
    end

    return GA
end

function get_backend_from_string(backend::String)
    backend_str = Unicode.normalize(backend, casefold=true)

    if backend_str == "cpu"
        backend = CPU()
    elseif backend_str == "cuda"
        backend = CUDABackend()
    elseif backend_str == "amdgpu"
        backend = ROCBackend()
    else
        error("Backend \"$(backend)\" not supported")
    end

    return backend
end

function get_floatT_from_string(float_type::String)
    float_type_str = Unicode.normalize(float_type, casefold=true)

    if float_type_str ∈ ("float32", "single")
        backend = CPU()
    elseif float_type_str ∈ ("float64", "double")
        backend = CUDABackend()
    else
        error("Float type \"$(float_type)\" not supported, only Float32 or Float64")
    end

    return backend
end

end
