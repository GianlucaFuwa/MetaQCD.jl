include("opes_kernel.jl")

mutable struct OPES <: AbstractBias
    is_first_step::Bool
    after_calculate::Bool

    is_static::Bool
    symmetric::Bool
    counter::Int64
    stride::Int64

    biasfactor::Float64
    bias_prefactor::Float64

    σ₀::Float64
    adaptive_σ::Bool
    adaptive_σ_stride::Int64
    adaptive_counter::Int64
    s̄::Float64 # running average
    S::Float64 # approx. bandwidth
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

    nker::Int64
    kernels::Vector{Kernel}
    nδker::Int64
    δkernels::Vector{Kernel}

    old_sum_weights::Float64
    old_Z::Float64
    old_KDEnorm::Float64

    function OPES(p::ParameterSet; verbose=Verbose1(), instance=1)
        println_verbose1(verbose, ">> Setting OPES instance $(instance)...")
        is_first_step = true
        after_calculate = false

        is_static = p.is_static
        symmetric = p.symmetric
        counter = 1
        stride = p.stride
        println_verbose1(verbose, "\t>> STRIDE = $(stride)")

        barrier = p.barrier
        println_verbose1(verbose, "\t>> BARRIER = $(barrier)")
        @assert barrier >= 0 "BARRIER must be > 0"
        biasfactor = p.biasfactor===nothing ? barrier : p.biasfactor
        println_verbose1(verbose, "\t>> BIASFACTOR = $(biasfactor)")
        @assert biasfactor > 1 "BIASFACTOR must be > 1"
        bias_prefactor = 1 - 1/biasfactor

        σ₀ = p.sigma0
        print_verbose1(verbose, "\t>> SIGMA0 = $(σ₀)")
        @assert σ₀ >= 0 "SIGMA0 must be >= 0"
        adaptive_σ = σ₀==0.0
        if adaptive_σ
            println_verbose1(verbose, " → ADAPTIVE_SIGMA enabled")
            adaptive_σ_stride = p.adaptive_σ_stride===nothing ? 10stride : p.adaptive_sigma_stride
            println_verbose1(verbose, "\t>> ADAPTIVE_SIGMA_STRIDE = $(adaptive_σ_stride)")
            @assert adaptive_σ_stride > 0 "ADAPTIVE_SIGMA_STRIDE must be > 0 if enabled"
            adaptive_counter = 0
        else
            println_verbose1(verbose, " → ADAPTIVE_SIGMA enabled")
            adaptive_σ_stride = p.adaptive_σ_stride
            println_verbose1(verbose, "\t>> ADAPTIVE_SIGMA_STRIDE = $(adaptive_σ_stride)")
            @assert adaptive_σ_stride > 0 "ADAPTIVE_SIGMA_STRIDE must be > 0 if enabled"
            adaptive_counter = 0
        end

        s̄ = 0.0
        S = 0.0
        σ_min = p.sigma_min
        print_verbose1(verbose, "\t>> SIGMA_MIN = $(σ_min)")
        @assert σ_min >= 0 "SIGMA_MIN must be > 0"
        fixed_σ = p.fixed_sigma
        print_verbose1(verbose, "\t>> FIXED_SIGMA = $(fixed_σ)")

        ϵ = p.opes_epsilon===nothing ? exp(-barrier/bias_prefactor) : p.opes_epsilon
        sum_weights = ϵ^bias_prefactor
        sum_weights² = sum_weights^2
        current_bias = 0.0
        no_Z = p.no_Z
        print_verbose1(verbose, "\t>> NO_Z = $(Z)")
        Z = 1.0
        KDEnorm = sum_weights

        threshold = p.threshold===nothing ? 1.0 : p.threshold
        print_verbose1(verbose, "\t>> THRESHOLD = $(threshold)")
        @assert threshold > 0 "THRESHOLD must be > 0"
        cutoff = p.cutoff===nothing ? sqrt(2barrier/bias_prefactor) : p.cutoff
        print_verbose1(verbose, "\t>> CUTOFF = $(cutoff)")
        @assert cutoff > 0 "CUTOFF must be > 0"
        cutoff² = cutoff²
        penalty = exp(-0.5cutoff²)

        nker = 0
        kernels = Vector{Kernel}(undef, 1000)
        nδker = 0
        δkernels = Vector{Kernel}(undef, 2)

        old_sum_weights = sum_weights
        old_Z = Z
        old_KDEnorm = KDEnorm

        return new(
            is_first_step, after_calculate,
            is_static, symmetric, counter, stride,
            biasfactor, bias_prefactor,
            σ₀, adaptive_σ, adaptive_σ_stride, adaptive_counter, s̄, S, σ_min, fixed_σ,
            ϵ, sum_weights, sum_weights², current_bias, no_Z, Z, KDEnorm,
            threshold, cutoff², penalty,
            nker, kernels, nδker, δkernels,
            old_sum_weights, old_Z, old_KDEnorm,
        )
    end
end

get_kernels(o::OPES) = o.kernels
get_δkernels(o::OPES) = o.δkernels
eachkernel(o::OPES) = view(o.kernels, 1:o.nker)
eachδkernel(o::OPES) = view(o.δkernels, 1:o.nδker)

function update!(o::OPES, cv, itrj)
    o.is_static && return nothing

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

    kernels = get_kernels(o)
    δkernels = get_δkernels(o)

    # update variance if adaptive_σ
    if o.adaptive_σ
        o.adaptive_counter += 1
        τ = o.adaptive_σ_stride
        if o.adaptive_counter < o.adaptive_σ_stride
            τ = o.adaptive_counter
        end
        diff = cv - o.s̄
        o.s̄ += diff / τ
        o.S += diff * (s-o.s̄)
        (o.adaptive_counter<o.adaptive_σ_stride && o.counter==1) && return nothing
    end

    itrj%o.stride != 0 && return nothing
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
    if adaptive_σ
        factor = o.biasfactor
        if o.counter == 2
            o.S *= factor
            o.σ₀ = sqrt(o.S / o.adaptive_counter / factor)
            o.σ₀ = max(o.σ₀, o.σ_min)
        end
        σ = sqrt(o.S / o.adaptive_counter / factor)
        σ = max(σ, o.σ_min)
    end

    if !o.fixed_σ
        s_rescaling = (3neff/4)^(-1/5)
        σ *= s_rescaling
        σ = max(σ, o.σ_min)
    end

    # height should be divided by sqrt(2π)*σ but this is cancelled out by Z
    # so we leave it out altogether but keep the s_rescaling
    height *= (o.σ₀/σ)

    # add new kernel
    add_kernel!(o, height, s, σ)

    # update Z
    if !o.no_Z
        # instead of redoing the whole summation, we add only the changes, knowing that
        # uprob = old_uprob + δ_uprob
        # and we also need to consider that in the new sum there are some new centers
        # and some disappeared ones
        sum_uprob = 0.0
        δsum_uprob = 0.0
        for kernel in eachkernel(o)
            for δkernel in eachδkernel(o)
                # take away contribution from kernels that are gone, and add new ones
                sgn = sign(δkernel.height)
                δsum_uprob += δkernel(kernel.center) + sgn*kernel(δkernel.center)
            end
        end
        for δkernel in eachδkernel(o)
            for δδkernel in eachδkernel(o)
                # now subtract the δ_uprob added before, but not needed
                sgn = sign(δkernel.height)
                δsum_uprob -= sgn*δδkernel(δkernel.center)
            end
        end

        sum_uprob = o.Z * o.old_KDEnorm * old_nker + δsum_uprob
        o.Z = sum_uprob / o.KDEnorm / o.nker
    end

    return nothing
end

function (o::OPES)(cv)
    calculate!(o, cv)
    return o.current_bias
end

function calculate!(o::OPES, cv)
    o.is_first_step && return nothing

    prob = 0.0
    for kernel in eachkernel(o)
        prob += kernel(cv)
    end
    prob /= o.KDEnorm

    current_bias = o.bias_prefactor * log(prob/o.Z + o.ϵ)
    o.current_bias = current_bias
    o.after_calculate = true
    return nothing
end

function ∂V∂Q(o::OPES, cv)
    deriv = 0.0
    for kernel in eachkernel(o)
        deriv += derivative(kernel, cv)
    end
    deriv /= o.KDEnorm
    return deriv
end

function add_kernel!(o::OPES, height, cv, σ)
    kernels = get_kernels(o)
    δkernels = get_δkernels(o)
    new_kernel = Kernel(height, cv, σ, o.cutoff², o.penalty)

    taker_i = get_mergeable_kernel(cv, kernels, o.threshold, o.nker)

    if taker_i < o.nker+1
        δkernels[1] = -1 * kernels[taker_i]
        kernels[taker_i] = kernels[taker_i] + new_kernel
        δkernels[2] = kernels[taker_i]
        o.nδker = 2
    else
        if o.nker+1 > length(kernels)
            push!(kernels, new_kernel)
            o.nker += 1
        else
            kernels[o.nker+1] = new_kernel
            o.nδker = 1
        end
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

const state_header = "# counter biasfactor sigma0 epsilon sum_weights " *
    "Z threshold cutoff penalty"
const kernel_header = "# height center sigma"

function write_to_file(o::OPES, filename)
    (tmppath, tmpio) = mktemp()
    println(tmpio, state_header)
    state_str = "$(o.counter)\t$(o.biasfactor)\t$(o.σ₀)\t$(o.ϵ)\t$(o.sum_weights)\t$(o.Z)" *
        "\t$(o.threshold)\t$(√o.cutoff²)\t$(o.penalty)"
    println(tmpio, state_str)
    println(tmpio, kernel_header)

    for kernel in eachkernel(o)
        kernel_str = "$(kernel.height)\t$(kernel.center)\t$(kernel.σ)"
        println(tmpio, kernel_str)
    end

    close(tmpio)
    mv(tmppath, filename*".txt", force=true)
    return nothing
end

function opes_from_file(p::ParameterSet, usebias)
    # state is stored in header, which is always read as a string so we have to parse it
    kernel_data, state_data = readdlm(usebias, comments=true, header=true)
    state_parse = [parse(Float64, state_param) for state_param in state_data]
    cutoff² = state_parse[end-1]
    penalty = state_parse[end]

    kernels = Vector{Kernel}(undef, size(kernels, 1))
    for i in axes(kernel_data, 1)
        kernels[i] = Kernel(kernel_data[i,:]..., cutoff², penalty)
    end

    return kernels, state_parse...
end
