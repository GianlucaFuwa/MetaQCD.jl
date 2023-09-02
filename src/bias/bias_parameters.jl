struct2dict(x::T) where {T} =
    Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))

abstract type BiasParameters end

Base.@kwdef mutable struct MetadynamicsParameters <: BiasParameters
    symmetric::Bool = true
    stride::Int64 = 1
    cvlims::NTuple{2, Float64} = (-7, 7)
    biasfactor::Float64 = Inf
    bin_width::Float64 = 1e-2
    weight::Float64 = 1e-2
    penalty_weight::Float64 = 1000
end

Base.@kwdef mutable struct OPESParameters <: BiasParameters
    symmetric::Bool = true
    counter::Int64 = 1
    stride::Int64 = 1
    # cvlims::NTuple{2, Float64} = (-7, 7)
    barrier::Float64 = 50
    biasfactor::Float64 = Inf
    bias_prefactor::Float64 = 1.0
    #
    σ₀::Float64 = 0.1
    adaptive_σ::Bool = (σ₀==0)
    adaptive_σ_stride::Int64 = 10stride
    adaptive_counter::Int64 = 0
    s̄::Float64 = 0.0
    S::Float64 = 0.0
    σ_min::Float64 = 1e-6
    fixed_σ::Bool = false
    #
    ϵ::Float64 = exp(-barrier/bias_prefactor)
    sum_weights::Float64 = ϵ^bias_prefactor
    sum_weights²::Float64 = sum_weights^2
    current_bias::Float64 = 0.0
    no_Z::Bool = false
    Z::Float64 = 1.0
    KDEnorm::Float64 = sum_weights
    #
    threshold::Float64 = 1.0
    cutoff²::Float64 = sqrt(2barrier/bias_prefactor)
    penalty::Float64 = exp(-0.5cutoff²)
end

function prepare_bias_from_dict(U, value_i::Dict)
    parameters = construct_bias_parameters_from_dict(value_i)
    return prepare_bias(U, parameters)
end

function prepare_bias(U, bias_parameters::T) where {T}
    if T == MetadynamicsParameters
        bias = MetadynamicsMeasurement(U, bias_parameters)
    elseif T == OPESParameters
        bias = OPESMeasurement(U, bias_parameters)
    else
        error(T, " is not supported in measurements")
    end

    return bias
end

const state_vars = [
    "counter",
    "biasfactor",
    "sigma0",
    "epsilon",
    "sum_weights",
    "Z",
    "threshold",
    "cutoff",
    "penalty",
]
