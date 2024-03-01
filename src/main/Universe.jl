module Universe

using Dates
using Unicode
using ..Utils
using ..Output

import ..Gaugefields: Gaugefield, CPU, CUDABackend, ROCBackend, WilsonGaugeAction
import ..Gaugefields: IwasakiGaugeAction, DBW2GaugeAction, SymanzikTreeGaugeAction
import ..Gaugefields: initial_gauges
import ..BiasModule: Bias
import ..Parameters: ParameterSet

struct Univ{TG,TB}
    U::TG
    bias::TB
    numinstances::Int64
end

function Univ(p::ParameterSet; use_mpi=false, fp=true)
    @level1("┌ Setting Universe...")
    NX, NY, NZ, NT = p.L
    β = p.beta
    initial = get_initial_from_string(p.initial)
    B = get_backend_from_string(p.backend)
    T = get_floatT_from_string(p.float_type)
    GA = get_gaugeaction_from_string(p.kind_of_gaction)

    @level1("|  L: $(NX)x$(NY)x$(NZ)x$(NT)")
    @level1("|  GaugeAction: $(GA)")
    @level1("|  beta: $β")
    @level1("└\n")

    if p.kind_of_bias != "none"
        if p.tempering_enabled && use_mpi==false
            numinstances = p.numinstances
            @level1("|  Using 1 + $(numinstances-1) instances\n")
            U₁ = initial_gauges(initial, NX, NY, NZ, NT, β, BACKEND=B, T=T, GA=GA)
            bias₁ = Bias(p, U₁; instance=0)

            U = Vector{typeof(U₁)}(undef, numinstances)
            bias = Vector{typeof(bias₁)}(undef, numinstances)

            bias[1] = bias₁
            U[1] = U₁
            for i in 2:numinstances
                U[i] = initial_gauges(initial, NX, NY, NZ, NT, β, BACKEND=B, T=T, GA=GA)
                bias[i] = Bias(p, U[i]; instance=i-1)
            end
        else
            numinstances = 1
            U = initial_gauges(initial, NX, NY, NZ, NT, β, BACKEND=B, T=T, GA=GA)
            bias = Bias(p, U)
        end
    else
        @assert p.tempering_enabled == false "tempering can only be enabled with bias"
        numinstances = 1
        U = initial_gauges(initial, NX, NY, NZ, NT, β, BACKEND=B, T=T, GA=GA)
        bias = nothing
    end

    return Univ{typeof(U), typeof(bias)}(U, bias, numinstances)
end

function get_backend_from_string(backend::String)
    backend_str = Unicode.normalize(backend, casefold=true)

    if backend_str == "cpu"
        backend = CPU
    elseif backend_str ∈ ("cuda", "cudabackend")
        backend = CUDABackend
    elseif backend_str ∈ ("roc", "rocbackend")
        backend = ROCBackend
    else
        error("Backend \"$(backend)\" not supported")
    end

    return backend
end

function get_floatT_from_string(float_type::String)
    float_type_str = Unicode.normalize(float_type, casefold=true)

    if float_type_str ∈ ("float32", "single")
        FloatT = Float32
    elseif float_type_str ∈ ("float64", "double")
        FloatT = Float64
    else
        error("Float type \"$(float_type)\" not supported, only Float32 or Float64")
    end

    return FloatT
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

end
