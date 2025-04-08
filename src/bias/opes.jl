include("opes_kernel.jl")

"""
    OPES <: AbstractBias

OPES bias-enhanced sampler from https://arxiv.org/abs/1909.07250 .

    OPES(p::OPESParameters; instance=1, dummy=false)

Create an instance of a OPES bias using the parameters given in `p`.

# Specifiable parameters
`kind_of_cv::String = "topcharge_clover"` - Collective variable
`numsmears_for_cv::Int64 = 4` - Number of smearing steps for the CV (step size is given in superstructure `Bias`)
`symmetric::Bool = true` - If `true`, the bias is built symmetrically by updating for both cv and
-cv at every update-iteration \\
`stride::Int64 = 1` - Number of iterations between updates; must be >0 \\
`cvlims::NTuple{2, Float64} = (-6, 6)` - Minimum and maximum of the explorable cv-space;
must be ordered \\
`write_bias_every::Int64 = 1` - Number of update iterations between writes of the bias to file \\
`barrier::Float64 = 30` - Estimate of height of action barriers \\
`biasfactor::Float64 = Inf` - Biasfactor for well-tempered OPES; must be >1 \\
`σ₀::Float64 = 0.1` - (Starting) width of kernels; must be >0 \\
`σ_min::Float64 = 1e-6` - Minimum width of kernels; must be >0 \\
`fixed_σ::Bool = true` - If `true`, width if kernels decreases iteratively \\
`ϵ::Float64 = exp(-barrier/(1-1/biasfactor))` - Determines maximum height of bias; must be >0 \\
`no_Z::Bool = false` - If `false` normalization factor `Z` is dynamically adjusted \\
`threshold::Float64 = 1.0` - Threshold distance for kernel merging; must be >0 \\
`cutoff::Float64 = sqrt(2barrier/(1-1/biasfactor))` - Cutoff value for kernels; must be >0 \\
`penalty::Float64 = exp(-0.5cutoff²)` - Penalty for being outside kernel cutoff; must be >0
"""
mutable struct OPES{CV} <: AbstractBias
    cvinfo::CV
    static::Bool
    is_first_step::Bool

    explore::Bool
    symmetric::Bool
    counter::Int64
    stride::Int64
    cvlims::NTuple{2,Float64}

    biasfactor::Float64
    bias_prefactor::Float64

    sigma0::Float64
    σ_min::Float64
    fixed_σ::Bool

    epsilon::Float64
    sum_weights::Float64
    sum_weights2::Float64
    current_bias::Float64
    current_weight::Float64
    no_Z::Bool
    Z::Float64
    KDEnorm::Float64

    threshold::Float64
    cutoff2::Float64
    penalty::Float64

    old_sum_weights::Float64
    old_Z::Float64
    old_KDEnorm::Float64

    nker::Int64
    kernels::Vector{Kernel}
    nδker::Int64
    δkernels::Vector{Kernel}
    write_bias_every::Int64
end

function OPES(p::OPESParameters; instance=1, dummy=false, build=false, mpi_multi_sim=false)
    inum = if dummy
        0
    elseif mpi_multi_sim
        MPI_INSTANCE[]
    else
        instance
    end

    cvinfo = get_cvinfo_from_parameters(p)
    static = if dummy
        true
    elseif build
        false
    else
        inum==0 ? false : p.static[inum]
    end
    is_first_step = true

    symmetric = p.symmetric
    explore = p.explore
    counter = 1
    stride = p.stride
    cvlims = tuple(p.cvlims...)

    barrier = p.barrier
    if explore
        @assert !isinf(p.biasfactor) "biasfactor has to be finite in explore!"
        biasfactor = p.biasfactor == 0.0 ? barrier : p.biasfactor
        bias_prefactor = biasfactor - 1
    else
        biasfactor = p.biasfactor == 0.0 ? barrier : p.biasfactor
        bias_prefactor = 1 - 1 / biasfactor
    end

    sigma0 = p.sigma_0
    σ_min = p.sigma_min
    fixed_σ = p.fixed_sigma

    epsilon = p.epsilon == 0.0 ? exp(-barrier / bias_prefactor) : p.epsilon
    sum_weights = epsilon^bias_prefactor
    sum_weights2 = sum_weights^2
    current_bias = 0.0
    no_Z = !p.adaptive_Z
    Z = 1.0

    threshold = p.threshold
    cutoff = if explore
        p.cutoff == 0.0 ? sqrt(2barrier) : p.cutoff
    else
        p.cutoff == 0.0 ? sqrt(2barrier / bias_prefactor) : p.cutoff
    end
    cutoff2 = cutoff^2
    penalty = exp(-0.5cutoff2)

    nker = 0
    kernels = Vector{Kernel}(undef, 0)
    nδker = 0
    δkernels = Vector{Kernel}(undef, 0)

    state = Dict{Symbol,Any}(
        :counter => counter,
        :biasfactor => biasfactor,
        :sigma0 => sigma0,
        :epsilon => epsilon,
        :sum_weights => sum_weights,
        :sum_weights2 => sum_weights2,
        :Z => Z,
        :threshold => threshold,
        :cutoff => cutoff,
        :penalty => penalty,
    )

    if (0 < instance <= length(p.usebiases) && !dummy) || (build && (length(p.usebiases) != 0))
        idx = build ? 1 : instance+1
        kernels, nker = opes_from_file!(state, p.usebiases[idx])
        is_first_step = false
        explore = state[:explore]
        counter = Int64(state[:counter])
        biasfactor = state[:biasfactor]
        bias_prefactor = 1 - 1 / biasfactor
        sigma0 = state[:sigma0]
        epsilon = state[:epsilon]
        sum_weights = state[:sum_weights]
        sum_weights2 = state[:sum_weights2]
        Z = state[:Z]
        threshold = state[:threshold]
        cutoff2 = state[:cutoff2]^2
        penalty = state[:penalty]
    end

    KDEnorm = explore ? counter : sum_weights

    write_bias_every = if p.write_bias_every <= stride
        stride
    else
        p.write_bias_every
    end

    @level1("|  STATIC: $(static)")
    @level1("|  EXPLORE: $(explore)")
    @level1("|  SYMMETRIC: $(symmetric)")
    @level1("|  NKER: $(nker)")
    @level1("|  COUNTER: $(counter)")
    @assert counter > 0 "COUNTER must be ≥0"
    @level1("|  STRIDE: $(stride)")
    @assert stride > 0 "STRIDE must be >0"
    @level1("|  CVLIMS: $(string(cvlims))")
    @assert issorted(p.cvlims) "CVLIMS must be sorted from low to high"
    @level1("|  BARRIER: $(barrier)")
    @assert barrier >= 0 "BARRIER must be > 0"
    @level1("|  BIASFACTOR: $(biasfactor)")
    @assert biasfactor > 1 "BIASFACTOR must be > 1"
    @level1("|  SIGMA0: $(sigma0)")
    @assert sigma0 >= 0 "SIGMA0 must be >= 0"
    @level1("|  SIGMA_MIN: $(σ_min)")
    @assert σ_min >= 0 "SIGMA_MIN must be > 0"
    @level1("|  FIXED_SIGMA: $(fixed_σ)")
    @level1("|  EPSILON: $(epsilon)")
    @assert epsilon > 0 "EPSILON must be > 0, maybe your BARRIER is to high?"
    @level1("|  NO_Z: $(no_Z)")
    @level1("|  THRESHOLD: $(threshold)")
    @assert threshold > 0 "THRESHOLD must be > 0"
    @level1("|  CUTOFF: $(sqrt(cutoff2))")
    @assert cutoff > 0 "CUTOFF must be > 0"
    @level1("|  WRITE_BIAS_EVERY: $(string(dummy ? "" : write_bias_every))")
    return OPES(
        cvinfo, static, is_first_step,
        explore, symmetric, counter, stride, cvlims,
        biasfactor, bias_prefactor,
        sigma0, σ_min, fixed_σ,
        epsilon, sum_weights, sum_weights2, current_bias, 0.0, no_Z, Z, KDEnorm,
        threshold, cutoff2, penalty,
        sum_weights, Z, KDEnorm,
        nker, kernels, nδker, δkernels, write_bias_every
    )
end

get_kernels(o::OPES) = o.kernels
get_δkernels(o::OPES) = o.δkernels
get_ext(::OPES) = ".opes"
is_adaptive(o::OPES) = (o.sigma0 == 0)
ext_length(::OPES) = Val(4)

function set_sigma0!(o::OPES, val)
    o.sigma0 = val
    @level1("|  sigma0 is adaptively set to $(o.sigma0)!")
    return nothing
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
    cutoff2 = o.cutoff2
    penalty = o.penalty

    prob = 0.0

    for kernel in o.kernels
        prob += kernel(cv, cutoff2, penalty)
        @assert prob < 1e10 "opes_prob = $prob is too high, something probably went wrong"
    end

    prob /= o.KDEnorm
    current_bias = o.bias_prefactor * log(prob / o.Z + o.epsilon)
    o.current_weight = prob
    o.current_bias = current_bias
    return nothing
end

function update!(o::OPES, cv, itrj)
    if o.is_first_step
        o.is_first_step = false
        return nothing
    end

    (itrj % o.stride != 0 || length(cv) == 0) && return nothing
    o.old_KDEnorm = o.KDEnorm
    old_nker = o.nker

    # if bias is symmetric then we add twice the weight to the total
    symm_factor = o.symmetric ? 2.0 : 1.0

    # get new kernel height
    current_bias = [o(cvᵢ) for cvᵢ in cv]
    height = [exp(Vᵢ) for Vᵢ in current_bias]

    # update sum_weights and neff
    o.counter += symm_factor * length(cv)
    o.sum_weights += symm_factor * sum(height)
    o.sum_weights2 += symm_factor^2 * sum(height .* height)
    neff = (1 + o.sum_weights)^2 / (1 + o.sum_weights2)
    o.KDEnorm = o.sum_weights

    # if needed rescale sigma and height
    σ = o.sigma0

    if !o.fixed_σ
        s_rescaling = (3neff / 4)^(-1 / 5)
        σ *= s_rescaling
        σ = max(σ, o.σ_min)
    end

    # height should be divided by sqrt(2π)*σ but this is cancelled out by Z
    # so we leave it out altogether but keep the s_rescaling
    height *= (o.sigma0 / σ)

    # add new kernels
    empty!(o.δkernels)
    o.nδker = 0
    for i in eachindex(cv)
        add_kernel!(o, height[i], cv[i], σ)
        o.symmetric && add_kernel!(o, height[i], -cv[i], σ)
    end

    # update Z
    if !o.no_Z
        # instead of redoing the whole summation, we add only the changes, knowing that
        # uprob = old_uprob + δ_uprob
        # and we also need to consider that in the new sum there are some new centers
        # and some disappeared ones
        cutoff2 = o.cutoff2
        penalty = o.penalty
        sum_uprob = 0.0
        δsum_uprob = 0.0
        for kernel in o.kernels
            for δkernel in o.δkernels
                # take away contribution from kernels that are gone, and add new ones
                sgn = sign(δkernel.height)
                δsum_uprob +=
                    δkernel(kernel.center, cutoff2, penalty) +
                    sgn * kernel(δkernel.center, cutoff2, penalty)
            end
        end
        for δkernel in o.δkernels
            for δδkernel in o.δkernels
                # now subtract the δ_uprob added before, but not needed
                sgn = sign(δkernel.height)
                δsum_uprob -= sgn * δδkernel(δkernel.center, cutoff2, penalty)
            end
        end

        sum_uprob = o.Z * o.old_KDEnorm * old_nker + δsum_uprob
        o.Z = sum_uprob / o.KDEnorm / o.nker
    end

    return nothing
end

function ∂V∂Q(o::OPES, cv)
    cutoff2 = o.cutoff2
    penalty = o.penalty

    prob = 0.0
    deriv = 0.0

    for kernel in o.kernels
        prob += kernel(cv, cutoff2, penalty)
        deriv += derivative(kernel, cv, cutoff2, penalty)
    end

    prob /= o.KDEnorm
    deriv /= o.KDEnorm

    Z = o.Z
    out = -o.bias_prefactor / (prob / Z + o.epsilon) * deriv / Z
    return out
end

function add_kernel!(o::OPES, height, cv, σ)
    kernels = get_kernels(o)
    δkernels = get_δkernels(o)
    new_kernel = Kernel(height, cv, σ)

    taker_i = get_mergeable_kernel(cv, kernels, o.threshold, o.nker)

    if taker_i < o.nker + 1
        push!(δkernels, -1 * kernels[taker_i])
        kernels[taker_i] = merge(kernels[taker_i], new_kernel)
        push!(δkernels, kernels[taker_i])
        o.nδker += 2
    else
        push!(kernels, new_kernel)
        o.nker += 1
        push!(δkernels, new_kernel)
        o.nδker += 1
    end

    return nothing
end

function get_mergeable_kernel(cv, kernels, threshold, nker)
    d_min = threshold
    taker_i = nker + 1

    for i in 1:nker
        d = abs(kernels[i].center - cv) / kernels[i].σ
        ismin = d < d_min
        d_min = ifelse(ismin, d, d_min)
        taker_i = ifelse(ismin, i, taker_i)
    end

    return taker_i
end

const opes_state_vars = [
    :counter,
    :biasfactor,
    :sigma0,
    :epsilon,
    :sum_weights,
    :sum_weights2,
    :KDEnorm,
    :Z,
    :threshold,
    :cutoff2,
    :penalty,
]

write_to_file(::OPES, ::Nothing) = nothing

function write_to_file(o::OPES, filename::AbstractString)
    filename=="" && return nothing
    (tmppath, tmpio) = mktemp()
    print(tmpio, "#")

    for var in opes_state_vars
        print(tmpio, rpad(var, 25))
    end

    println(tmpio)

    for var in opes_state_vars
        print(tmpio, rpad(getproperty(o, var), 25))
    end

    println(tmpio, "\n")
    print(tmpio, rpad("#height", 25))
    print(tmpio, rpad("center", 25))
    print(tmpio, rpad("sigma", 25))

    println(tmpio)

    for kernel in o.kernels
        @printf(tmpio, "%+-25.15e", kernel.height)
        @printf(tmpio, "%+-25.15e", kernel.center)
        @printf(tmpio, "%+-25.15e", kernel.σ)
        println(tmpio)
    end

    close(tmpio)
    mv(tmppath, filename; force=true)
    return nothing
end

function opes_from_file!(dict, usebias)
    if usebias == ""
        kernels = Vector{Kernel}(undef, 0)
        return kernels, 0
    else
        @level1("|  Getting state from $(usebias)")
        # state is stored in header, which is always read as a string so we have to parse it
        kernel_data, state_data = readdlm(usebias; comments=true, header=true)
        state_parse = [parse(Float64, state_param) for state_param in state_data]

        for i in eachindex(opes_state_vars)
            dict[opes_state_vars[i]] = state_parse[i]
        end

        kernels = Vector{Kernel}(undef, size(kernel_data, 1))

        for i in axes(kernel_data, 1)
            kernels[i] = Kernel(view(kernel_data, i, 1:3)...)
        end

        return kernels, length(kernels)
    end
end
