include("opes_kernel.jl")

"""
    OPES(p::ParameterSet; verbose=Verbose1(), instance=1)

Create an instance of a OPES bias using the parameters given in `p`. \\
`verbose` is used to print all parameters out as they are explicitly defined in the
constructor.

# Specifiable parameters
`symmetric::Bool = true` - If `true`, the bias is built symmetrically by updating for both cv and
-cv at every update-iteration \\
`stride::Int64 = 1` - Number of iterations between updates; must be >0 \\
`cvlims::NTuple{2, Float64} = (-6, 6)` - Minimum and maximum of the explorable cv-space;
must be ordered \\
`barrier::Float64 = 30` - Estimate of height of action barriers \\
`biasfactor::Float64 = Inf` - Biasfactor for well-tempered OPES; must be >1 \\
`σ₀::Float64 = 0.1` - (Starting) width of kernels; must be >0 \\
`σ_min::Float64 = 1e-6` - Minimum width of kernels; must be >0 \\
`fixed_σ::Bool = true` - If `true`, width if kernels decreases iteratively \\
`ϵ::Float64 = exp(-barrier/(1-1/biasfactor))` - Determines maximum height of bias; must be >0 \\
`no_Z::Bool = false` - If `false` normalization factor `Z` is dynamically adjusted \\
`threshold::Float64 = 1.0` - Threshold distance for kernel merging; must be >0 \\
`cutoff::Float64 = sqrt(2barrier/(1-1/biasfactor))` - Cutoff value for kernels; must be >0 \\
`penalty::Float64 = exp(-0.5cutoff²)` - Penalty for being outside kernel cutoff; must be >0 \\
"""
mutable struct OPES <: AbstractBias
    is_first_step::Bool
    after_calculate::Bool

    symmetric::Bool
    counter::Int64
    stride::Int64
    cvlims::NTuple{2, Float64}

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

function OPES(p::ParameterSet; instance=1)
    is_first_step = true
    after_calculate = false

    symmetric = p.symmetric
    counter = 1
    stride = p.stride
    cvlims = p.cvlims

    barrier = p.barrier
    biasfactor = p.biasfactor===0.0 ? barrier : p.biasfactor
    bias_prefactor = 1 - 1/biasfactor

    σ₀ = p.sigma0
    σ_min = p.sigma_min
    fixed_σ = p.fixed_sigma

    ϵ = p.opes_epsilon===0.0 ? exp(-barrier/bias_prefactor) : p.opes_epsilon
    sum_weights = ϵ^bias_prefactor
    sum_weights² = sum_weights^2
    current_bias = 0.0
    no_Z = p.no_Z
    Z = 1.0
    KDEnorm = sum_weights

    threshold = p.threshold
    cutoff = p.cutoff===0.0 ? sqrt(2barrier/bias_prefactor) : p.cutoff
    cutoff² = cutoff^2
    penalty = exp(-0.5cutoff²)

    nker = 0
    kernels = Vector{Kernel}(undef, 1000)
    nδker = 0
    δkernels = Vector{Kernel}(undef, 2)

    state = Dict{String, Any}(
        "counter" => counter,
        "biasfactor" => biasfactor,
        "sigma0" => σ₀,
        "epsilon" => ϵ,
        "sum_weights" => sum_weights,
        "Z" => Z,
        "threshold" => threshold,
        "cutoff" => cutoff,
        "penalty" => penalty,
    )

    if 0<instance<=length(p.usebiases)
        kernels, nker = opes_from_file!(state, p.usebiases[instance])
        is_first_step = false
        counter = Int64(state["counter"])
        biasfactor = state["biasfactor"]
        σ₀ = state["sigma0"]
        ϵ = state["epsilon"]
        sum_weights = state["sum_weights"]
        Z = state["Z"]
        threshold = state["threshold"]
        cutoff² = state["cutoff"]^2
        penalty = state["penalty"]
    end

    @level1("|  NKER: $(nker)")
    @level1("|  COUNTER: $(counter)")
    @assert counter>0 "COUNTER must be ≥0"
    @level1("|  STRIDE: $(stride)")
    @assert stride>0 "STRIDE must be >0"
    @level1("|  CVLIMS: $(cvlims)")
    @assert cvlims[1]<cvlims[2] "CVLIMS[1] must be <CVLIMS[2]"
    @level1("|  BARRIER: $(barrier)")
    @assert barrier >= 0 "BARRIER must be > 0"
    @level1("|  BIASFACTOR: $(biasfactor)")
    @assert biasfactor > 1 "BIASFACTOR must be > 1"
    @level1("|  SIGMA0: $(σ₀)")
    @assert σ₀ >= 0 "SIGMA0 must be >= 0"
    @level1("|  SIGMA_MIN: $(σ_min)")
    @assert σ_min >= 0 "SIGMA_MIN must be > 0"
    @level1("|  FIXED_SIGMA: $(fixed_σ)")
    @level1("|  EPSILON: $(ϵ)")
    @assert ϵ > 0 "EPSILON must be > 0, maybe your BARRIER is to high?"
    @level1("|  NO_Z: $(no_Z)")
    @level1("|  THRESHOLD: $(threshold)")
    @assert threshold > 0 "THRESHOLD must be > 0"
    @level1("|  CUTOFF: $(sqrt(cutoff²))")
    @assert cutoff > 0 "CUTOFF must be > 0"
    return OPES(is_first_step, after_calculate,
                symmetric, counter, stride, cvlims,
                biasfactor, bias_prefactor,
                σ₀, σ_min, fixed_σ,
                ϵ, sum_weights, sum_weights², current_bias, 0.0, no_Z, Z, KDEnorm,
                threshold, cutoff², penalty,
                sum_weights, Z, KDEnorm,
                nker, kernels, nδker, δkernels)
end

get_kernels(o::OPES) = o.kernels
get_δkernels(o::OPES) = o.δkernels
eachkernel(o::OPES) = view(o.kernels, 1:o.nker)
eachδkernel(o::OPES) = view(o.δkernels, 1:o.nδker)

function update!(o::OPES, cv, itrj)
    calculate!(o, cv)
    update_opes!(o, cv, itrj)

    if o.symmetric
        calculate!(o, -cv)
        update_opes!(o, -cv, itrj)
    end

    return nothing
end

function update_opes!(o::OPES, cv, itrj)
    if o.is_first_step
        o.is_first_step = false
        return nothing
    end

    (itrj%o.stride != 0 || !in_bounds(cv, o.cvlims...)) && return nothing
    o.old_KDEnorm = o.KDEnorm
    old_nker = o.nker

    # get new kernel height
    height = exp(o.current_bias)

    # update sum_weights and neff
    o.counter += 1
    o.sum_weights += height
    o.sum_weights² += height^2
    neff = (1 + o.sum_weights)^2 / (1 + o.sum_weights²)
    o.KDEnorm = o.sum_weights

    # if needed rescale sigma and height
    σ = o.σ₀

    if !o.fixed_σ
        s_rescaling = (3neff/4)^(-1/5)
        σ *= s_rescaling
        σ = max(σ, o.σ_min)
    end

    # height should be divided by sqrt(2π)*σ but this is cancelled out by Z
    # so we leave it out altogether but keep the s_rescaling
    height *= (o.σ₀/σ)

    # add new kernel
    add_kernel!(o, height, cv, σ)

    # update Z
    if !o.no_Z
        # instead of redoing the whole summation, we add only the changes, knowing that
        # uprob = old_uprob + δ_uprob
        # and we also need to consider that in the new sum there are some new centers
        # and some disappeared ones
        cutoff² = o.cutoff²
        penalty = o.penalty
        sum_uprob = 0.0
        δsum_uprob = 0.0
        for kernel in eachkernel(o)
            for δkernel in eachδkernel(o)
                # take away contribution from kernels that are gone, and add new ones
                sgn = sign(δkernel.height)
                δsum_uprob += δkernel(kernel.center, cutoff², penalty) +
                              sgn*kernel(δkernel.center, cutoff², penalty)
            end
        end
        for δkernel in eachδkernel(o)
            for δδkernel in eachδkernel(o)
                # now subtract the δ_uprob added before, but not needed
                sgn = sign(δkernel.height)
                δsum_uprob -= sgn*δδkernel(δkernel.center, cutoff², penalty)
            end
        end

        sum_uprob = o.Z * o.old_KDEnorm * old_nker + δsum_uprob
        o.Z = sum_uprob / o.KDEnorm / o.nker
    end

    return nothing
end

function (o::OPES)(cv)
    lb, ub = o.cvlims

    if !in_bounds(cv, lb, ub)
        bounds_penalty = 100
        which_bound, dist² = findmin(((cv - lb)^2, (cv - ub)^2))
        nearest_bound = which_bound==1 ? lb : ub
        calculate!(o, nearest_bound)
        return o.current_bias + bounds_penalty*dist²
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
    for kernel in eachkernel(o)
        prob += kernel(cv, cutoff², penalty)
        if prob > 1e10
            throw(AssertionError("prob = $prob is dangerously high,
                                  something probably went wrong"))
        end
    end
    prob /= o.sum_weights

    current_bias = o.bias_prefactor * log(prob/o.Z + o.ϵ)
    o.current_weight = prob
    o.current_bias = current_bias
    o.after_calculate = true
    return nothing
end

function ∂V∂Q(o::OPES, cv)
    cutoff² = o.cutoff²
    penalty = o.penalty

    prob = 0.0
    deriv = 0.0
    for kernel in eachkernel(o)
        prob += kernel(cv, cutoff², penalty)
        deriv += derivative(kernel, cv, cutoff², penalty)
    end
    prob /= o.sum_weights
    deriv /= o.sum_weights

    Z = o.Z
    out = -o.bias_prefactor / (prob/Z+o.ϵ) * deriv/Z
    return out
end

function add_kernel!(o::OPES, height, cv, σ)
    kernels = get_kernels(o)
    δkernels = get_δkernels(o)
    new_kernel = Kernel(height, cv, σ)

    taker_i = get_mergeable_kernel(cv, kernels, o.threshold, o.nker)

    if taker_i < o.nker+1
        δkernels[1] = -1 * kernels[taker_i]
        kernels[taker_i] = merge(kernels[taker_i], new_kernel)
        δkernels[2] = kernels[taker_i]
        o.nδker = 2
    else
        if o.nker+1 > length(kernels)
            push!(kernels, new_kernel)
            o.nker += 1
        else
            kernels[o.nker+1] = new_kernel
            o.nker += 1
        end
        δkernels[1] = new_kernel
        o.nδker = 1
    end

    return nothing
end

function get_mergeable_kernel(cv, kernels, threshold, nker)
    d_min = threshold
    taker_i = nker + 1

    for i in 1:nker
        d = abs(kernels[i].center-cv) / kernels[i].σ
        ismin = d < d_min
        d_min = ifelse(ismin, d, d_min)
        taker_i = ifelse(ismin, i, taker_i)
    end

    return taker_i
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
const kernel_header = "#$(rpad("height", 20))\t$(rpad("center", 20))\t$(rpad("sigma", 20))"

write_to_file(::OPES, ::Nothing) = nothing

function write_to_file(o::OPES, filename::String)
    filename=="" && return nothing
    (tmppath, tmpio) = mktemp()
    print(tmpio, "#"); [print(tmpio, "$(var)\t") for var in state_vars]; println(tmpio, "")
    state_str = "$(o.counter)\t$(o.biasfactor)\t$(o.σ₀)\t$(o.ϵ)\t$(o.sum_weights)\t$(o.Z)" *
        "\t$(o.threshold)\t$(√o.cutoff²)\t$(o.penalty)\n"
    println(tmpio, state_str)
    println(tmpio, kernel_header)

    for kernel in eachkernel(o)
        h_str = @sprintf("%+-20.15e", kernel.height)
        c_str = @sprintf("%+-20.15e", kernel.center)
        σ_str = @sprintf("%+-20.15e", kernel.σ)
        kernel_str = "$(h_str)\t$(c_str)\t$(σ_str)"
        println(tmpio, kernel_str)
    end

    close(tmpio)
    mv(tmppath, filename, force=true)
    return nothing
end

function opes_from_file!(dict, usebias)
    if usebias==""
        kernels = Vector{Kernel}(undef, 1000)
        return kernels, 0
    else
        @level1("|  Getting state from $(usebias)")
        # state is stored in header, which is always read as a string so we have to parse it
        kernel_data, state_data = readdlm(usebias, comments=true, header=true)
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
