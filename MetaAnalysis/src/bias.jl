# INFO: Copy the definition of the Metadynamics and OPES struct from
# MetaQCD so that we dont have to add it as a dependency and cut loading time

@inline function in_bounds(cv, lb, ub)
    lb <= cv < ub && return true
    return false
end

struct Metadynamics
    symmetric::Bool
    stride::Int64
    cvlims::NTuple{2,Float64}

    biasfactor::Float64
    bin_width::Float64
    weight::Float64
    penalty_weight::Float64

    bin_vals::Vector{Float64}
    values::Vector{Float64}
end

Base.length(m::Metadynamics) = length(m.values)
Base.eachindex(m::Metadynamics) = eachindex(m.values)
Base.lastindex(m::Metadynamics) = lastindex(m.values)

@inline function Base.getindex(m::Metadynamics, i)
    return m.values[i]
end

@inline function index(m::Metadynamics, cv)
    idx = (cv - m.cvlims[1]) / m.bin_width + 0.5
    return round(Int64, idx, RoundNearestTiesAway)
end

(m::Metadynamics)(cv) = return_potential(m, cv)

function return_potential(m::Metadynamics, cv)
    bw = m.bin_width
    pen = m.penalty_weight
    lb, ub = m.cvlims

    if in_bounds(cv, lb, ub)
        idx = index(m, cv)
        interpolation_constant = (cv - m.bin_vals[idx]) / bw
        return m[idx] * (1 - interpolation_constant) + interpolation_constant * m[idx + 1]
    elseif cv < lb
        penalty = m[1] + pen * (cv - lb)^2
        return penalty
    else
        penalty = m[end] + pen * (cv - ub)^2
        return penalty
    end
end

function ∂V∂Q(m::Metadynamics, cv)
    bw = m.bin_width
    num = -m(cv + 2bw) + 8m(cv + bw) - 8m(cv - bw) + m(cv - 2bw)
    denom = 12bw
    return num / denom
end

struct Kernel
    height::Float64
    center::Float64
    σ::Float64
end

mutable struct OPES
    is_first_step::Bool

    symmetric::Bool
    counter::Int64
    stride::Int64
    cvlims::NTuple{2,Float64}

    biasfactor::Float64
    bias_prefactor::Float64

    σ₀::Float64
    σ_min::Float64
    fixed_σ::Bool

    ϵ::Float64
    sum_weights::Float64
    sum_weights²::Float64
    current_bias::Float64
    current_weight::Float64
    no_Z::Bool
    Z::Float64
    KDEnorm::Float64

    threshold::Float64
    cutoff²::Float64
    penalty::Float64

    old_sum_weights::Float64
    old_Z::Float64
    old_KDEnorm::Float64

    nker::Int64
    kernels::Vector{Kernel}
    nδker::Int64
    δkernels::Vector{Kernel}
end

function OPES(filename::String)
    state = Dict{String,Any}(
        "counter" => 0,
        "biasfactor" => Inf,
        "sigma0" => 0.0,
        "epsilon" => 0.0,
        "sum_weights" => 0.0,
        "Z" => 1.0,
        "threshold" => 1.0,
        "cutoff" => 1.0,
        "penalty" => 1.0,
    )

    @assert isfile(filename) "file \"$(filename)\" doesn't exist"
    kernels, nker = opes_from_file!(state, filename)
    is_first_step = false
    counter = Int64(state["counter"])
    biasfactor = state["biasfactor"]
    bias_prefactor = 1 - 1 / biasfactor
    σ₀ = state["sigma0"]
    ϵ = state["epsilon"]
    sum_weights = state["sum_weights"]
    Z = state["Z"]
    threshold = state["threshold"]
    cutoff² = state["cutoff"]^2
    penalty = state["penalty"]

    return OPES(
        is_first_step,
        true,
        counter,
        1,
        (-5, 5),
        biasfactor,
        bias_prefactor,
        σ₀,
        1e-6,
        false,
        ϵ,
        sum_weights,
        sum_weights^2,
        0.0,
        0.0,
        false,
        Z,
        sum_weights,
        threshold,
        cutoff²,
        penalty,
        sum_weights,
        Z,
        sum_weights,
        nker,
        kernels,
        0,
        Vector{Kernel}(undef, 2),
    )
end

function (o::OPES)(cv)
    lb, ub = o.cvlims

    if !in_bounds(cv, lb, ub)
        bounds_penalty = 100
        which_bound, dist² = findmin(((cv - lb)^2, (cv - ub)^2))
        nearest_bound = which_bound == 1 ? lb : ub
        calculate!(o, nearest_bound)
        return o.current_bias + bounds_penalty * dist²
    else
        calculate!(o, cv)
        return o.current_bias
    end
end

function calculate!(o::OPES, cv)
    o.is_first_step && return nothing
    cutoff² = o.cutoff²
    penalty = o.penalty

    prob = 0.0
    for kernel in o.kernels
        prob += kernel(cv, cutoff², penalty)
        if prob > 1e10
            throw(AssertionError("prob = $prob is dangerously high,
                                  something probably went wrong"))
        end
    end
    prob /= o.sum_weights

    current_bias = o.bias_prefactor * log(prob / o.Z + o.ϵ)
    o.current_weight = prob
    o.current_bias = current_bias
    return nothing
end

function ∂V∂Q(o::OPES, cv)
    cutoff² = o.cutoff²
    penalty = o.penalty

    prob = 0.0
    deriv = 0.0
    for kernel in o.kernels
        prob += kernel(cv, cutoff², penalty)
        deriv += derivative(kernel, cv, cutoff², penalty)
    end
    prob /= o.sum_weights
    deriv /= o.sum_weights

    Z = o.Z
    out = -o.bias_prefactor / (prob / Z + o.ϵ) * deriv / Z
    return out
end

const state_vars = [
    "counter",
    "biasfactor",
    "sigma0",
    "epsilon",
    "sum_weights",
    "sum_weights²",
    "Z",
    "threshold",
    "cutoff",
    "penalty",
]

function opes_from_file!(dict, usebias)
    if usebias == ""
        kernels = Vector{Kernel}(undef, 0)
        return kernels, 0
    else
        # state is stored in header, which is always read as a string so we have to parse it
        kernel_data, state_data = readdlm(usebias; comments=true, header=true)
        state_parse = [parse(Float64, state_param) for state_param in state_data]

        for i in eachindex(state_vars)
            dict[state_vars[i]] = state_parse[i]
        end

        kernels = Vector{Kernel}(undef, size(kernel_data, 1))
        for i in axes(kernel_data, 1)
            kernels[i] = Kernel(view(kernel_data, i, 1:3)...)
        end
        return kernels, length(kernels)
    end
end

function (k::Kernel)(s, cutoff², penalty)
    return evaluate_kernel(s, k.height, k.center, k.σ, cutoff², penalty)
end

@inline function evaluate_kernel(s, height, center, σ, cutoff², penalty)
    diff = (center - s) / σ
    diff² = diff^2
    out = ifelse(diff² >= cutoff², 0.0, height * (exp(-0.5diff²) - penalty))
    return out
end

function derivative(k::Kernel, s, cutoff², penalty)
    return kernel_derivative(s, k.height, k.center, k.σ, cutoff², penalty)
end

@inline function kernel_derivative(s, height, center, σ, cutoff², penalty)
    diff = (center - s) / σ
    diff² = diff^2
    val = ifelse(diff² >= cutoff², 0.0, height * (exp(-0.5diff²) - penalty))
    out = -diff / σ * val
    return out
end

Base.:*(c::Real, k::Kernel) = Kernel(c * k.height, k.center, k.σ)

function merge(k::Kernel, other::Kernel) # Kernel merger
    h = k.height + other.height
    c = (k.height * k.center + other.height * other.center) / h
    s_my_part = k.height * (k.σ^2 + k.center^2)
    s_other_part = other.height * (other.σ^2 + other.center^2)
    s² = (s_my_part + s_other_part) / h - c^2
    return Kernel(h, c, sqrt(s²))
end
