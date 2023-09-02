include("opes_kernel.jl")

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

function OPES(p::ParameterSet; verbose=Verbose1(), instance=1)
    println_verbose1(verbose, ">> Setting OPES instance $(instance)...")
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
    if (p.usebiases!==nothing) && (0<instance<=length(p.usebiases))
        println_verbose1(verbose, "\t>> Getting state from $(p.usebiases[instance])")
        kernels, state = opes_from_file(p.usebiases[instance])
        nker = length(kernels)
        println_verbose1(verbose, "\t>> NKER = $(nker)")
        println_verbose1(verbose, "\t>> COUNTER = $(counter)")
        is_first_step = false
        counter = Int64(state["counter "])
        biasfactor = state["biasfactor "]
        σ₀ = state["sigma0 "]
        ϵ = state["epsilon "]
        sum_weights = state["sum_weights "]
        Z = state["Z "]
        threshold = state["threshold "]
        cutoff² = state["cutoff "]^2
        penalty = state["penalty "]
    end

    println_verbose1(verbose, "\t>> STRIDE = $(stride)")
    @assert stride>0 "STRIDE must be >0"
    println_verbose1(verbose, "\t>> CVLIMS = $(cvlims)")
    @assert cvlims[1]<cvlims[2] "CVLIMS[1] must be <CVLIMS[2]"
    println_verbose1(verbose, "\t>> BARRIER = $(barrier)")
    @assert barrier >= 0 "BARRIER must be > 0"
    println_verbose1(verbose, "\t>> BIASFACTOR = $(biasfactor)")
    @assert biasfactor > 1 "BIASFACTOR must be > 1"
    println_verbose1(verbose, "\t>> SIGMA0 = $(σ₀)")
    @assert σ₀ >= 0 "SIGMA0 must be >= 0"
    println_verbose1(verbose, "\t>> SIGMA_MIN = $(σ_min)")
    @assert σ_min >= 0 "SIGMA_MIN must be > 0"
    println_verbose1(verbose, "\t>> FIXED_SIGMA = $(fixed_σ)")
    println_verbose1(verbose, "\t>> EPSILON = $(ϵ)")
    @assert ϵ > 0 "EPSILON must be > 0, maybe your BARRIER is to high?"
    println_verbose1(verbose, "\t>> NO_Z = $(no_Z)")
    println_verbose1(verbose, "\t>> THRESHOLD = $(threshold)")
    @assert threshold > 0 "THRESHOLD must be > 0"
    println_verbose1(verbose, "\t>> CUTOFF = $(cutoff)")
    @assert cutoff > 0 "CUTOFF must be > 0"

    return OPES(
        is_first_step, after_calculate,
        symmetric, counter, stride, cvlims,
        biasfactor, bias_prefactor,
        σ₀, σ_min, fixed_σ,
        ϵ, sum_weights, sum_weights², current_bias, no_Z, Z, KDEnorm,
        threshold, cutoff², penalty,
        sum_weights, Z, KDEnorm,
        nker, kernels, nδker, δkernels,
    )
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
        bounds_penalty = 1000
        which_bound, dist² = findmin((cv - lb)^2, (cv - ub)^2)
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
            error("prob = $prob is dangerously high, something probably went wrong")
        end
    end
    prob /= o.sum_weights

    current_bias = o.bias_prefactor * log(prob/o.Z + o.ϵ)
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
    out = o.bias_prefactor / (prob/Z+o.ϵ) * deriv/Z
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
    "counter ",
    "biasfactor ",
    "sigma0 ",
    "epsilon ",
    "sum_weights ",
    "Z ",
    "threshold ",
    "cutoff ",
    "penalty ",
]
const kernel_vars = [
    "height ",
    "center ",
    "sigma ",
]

function write_to_file(o::OPES, filename)
    (tmppath, tmpio) = mktemp()
    println(tmpio, "# ", state_vars...)
    state_str = "$(o.counter) $(o.biasfactor) $(o.σ₀) $(o.ϵ) $(o.sum_weights) $(o.Z)" *
        " $(o.threshold) $(√o.cutoff²) $(o.penalty)"
    println(tmpio, state_str)
    println(tmpio, "# ", kernel_vars...)

    for kernel in eachkernel(o)
        kernel_str = "$(kernel.height) $(kernel.center) $(kernel.σ)"
        println(tmpio, kernel_str)
    end

    close(tmpio)
    mv(tmppath, filename, force=true)
    return nothing
end

function opes_from_file(usebias)
    usebias === nothing && return nothing, nothing
    # state is stored in header, which is always read as a string so we have to parse it
    kernel_data, state_data = readdlm(usebias, comments=true, header=true)
    state_parse = [parse(Float64, state_param) for state_param in state_data]
    state_dict =
        Dict{String, Any}(state_vars[i] => state_parse[i] for i in eachindex(state_vars))

    kernels = Vector{Kernel}(undef, size(kernel_data, 1))
    for i in axes(kernel_data, 1)
        kernels[i] = Kernel(view(kernel_data, i, 1:3)...)
    end

    return kernels, state_dict
end
