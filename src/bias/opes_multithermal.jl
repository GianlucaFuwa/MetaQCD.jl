"""
    OPESmultithermal{CV} <: AbstractBias

OPES bias-enhanced sampler from https://arxiv.org/abs/1909.07250 .

    OPESmultithermal(; symmetric=true, stride=1, cvlims=(-6, 6), barrier=30,
         biasfactor=Inf, σ₀=0.1, σ_min=1e-6, fixed_σ=true, opes_epsilon=0.0,
         no_Z=false, threshold=1.0, cutoff=0.0)
    OPESmultithermal(p::OPESmultithermalParameters; dummy=false)

Create an instance of a OPES bias using the parameters given in `p`.

# Specifiable parameters
`numsmears_for_cv::Int64 = 4` - Number of smearing steps for the CV (step size is given in superstructure `Bias`)
`symmetric::Bool = true` - If `true`, the bias is built symmetrically by updating for both cv and
-cv at every update-iteration \\
`stride::Int64 = 1` - Number of iterations between updates; must be >0 \\
`write_bias_every::Int64 = 1` - Number of update iterations between writes of the bias to file \\
`beta_min_max::Vector{Float64} = [6.0, 6.3]` - Minimum and maximum of beta range (must be ordered) \\
`beta_num::Int64 = 10` - Number of intermediate betas between `beta_min` and `beta_max` \\
"""
mutable struct OPESmultithermal{CV} <: AbstractBias
    cvinfo::CV
    static::Bool
    is_first_step::Bool
    stride::Int64
    beta0::Float64
    counter::Int64
    rct::Float64
    current_bias::Float64
    current_weight::Vector{Float64}

    β_min::Float64
    β_max::Float64
    β::Vector{Float64}
    λ::Vector{Float64}
    ΔF::Vector{Float64}
    sum_weights::Vector{Float64}
    sum_weights2::Vector{Float64}
    write_bias_every::Int64
end

function OPESmultithermal(
    p::OPESmultithermalParameters, beta0;
    instance=1, dummy=false, mpi_multi_sim=false, build=false
)
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
    stride = p.stride
    counter = 1
    rct = 0.0

    @assert length(p.beta_min_max) == 2 "beta_min_max has to be a vector of length 2"
    β_min = minimum(p.beta_min_max)
    β_max = maximum(p.beta_min_max)
    @assert p.beta_num > 0 "beta_num has to be bigger than 0"
    β = collect(sort(range(β_min, β_max, p.beta_num), rev=true))
    λ = dummy ? zero(β) : (β .- beta0) / beta0
    ΔF = zero(λ)
    sum_weights = zero(λ)
    sum_weights2 = zero(λ)
    current_weight = zero(λ)

    if (0 < instance <= length(p.load_bias) && !dummy)
        load_bias = p.load_bias[instance+1]
        is_first_step = false
        counter, rct, β, λ, ΔF = opesmt_from_file!(counter, rct, β, λ, ΔF, load_bias)
    elseif build && (length(p.load_bias) != 0)
        load_bias = p.load_bias[1]
        is_first_step = false
        counter, rct, β, λ, ΔF = opesmt_from_file!(counter, rct, β, λ, ΔF, load_bias)
    end
    
    write_bias_every = if p.write_bias_every <= stride
        stride
    else
        p.write_bias_every
    end

    @level1("|  STATIC: $(static)")
    @level1("|  STRIDE: $(stride)")
    @assert stride > 0 "STRIDE must be >0"
    @level1("|  MIN BETA: $(β_min)")
    @level1("|  MAX BETA: $(β_max)")
    @level1("|  NUM INTERMEDIATES: $(p.beta_num)")
    @level1("|  WRITE_BIAS_EVERY: $(string(dummy ? "" : write_bias_every))")
    return OPESmultithermal(
        cvinfo, static, is_first_step, stride, beta0, counter, rct, 0.0, current_weight,
        β_min, β_max, β, λ, ΔF, sum_weights, sum_weights2, write_bias_every
    )
end

get_ext(::OPESmultithermal) = ".opesmt"
is_adaptive(o::OPESmultithermal) = false
set_sigma0!(::OPESmultithermal, ::Any) = nothing
ext_length(::OPESmultithermal) = Val(6)

function (o::OPESmultithermal)(cv)
    calculate!(o, cv)
    return o.current_bias
end

function calculate!(o::OPESmultithermal, cv::Float64)
    ΔF = o.ΔF
    λ = o.λ
    ΔS_max = maximum(-cv*λ[i] + ΔF[i] for i in eachindex(λ)) # get maximum difference to avoid over/underflow of exp
    sum = 0.0

    for i in eachindex(λ)
        diff_i = -cv*λ[i] + ΔF[i]
        sum += exp(diff_i - ΔS_max)
    end

    for i in eachindex(λ)
        o.current_weight[i] = exp(-ΔS_max) - sum/length(λ)
    end

    current_bias = -(ΔS_max + log(sum/length(λ)))
    o.current_bias = current_bias
    return nothing
end

function update!(o::OPESmultithermal, cv, itrj)
    if o.is_first_step
        o.is_first_step = false
        return nothing
    end

    (itrj % o.stride != 0) && return nothing

    for i in eachindex(cv)
        current_bias = o(cv[i])
        update_ΔF!(o, current_bias, cv[i])
    end

    return nothing
end

function update_ΔF!(o::OPESmultithermal, current_bias::Float64, cv::Float64)
    o.counter += 1
    counter = o.counter
    rct = o.rct
    arg = current_bias - rct - log(counter-1)
    λ = o.λ
    ΔF = o.ΔF
    sum_weights = o.sum_weights
    increment = arg > 0 ? arg + log1p(exp(-arg)) : log1p(exp(arg)) # save exp from overflow

    for i in eachindex(λ)
        ΔS = λ[i] * cv
        diff = -ΔS + current_bias - rct + ΔF[i] - log1p(counter - 1)

        if diff > 0 # save exp from overflow
            ΔF[i] += increment - diff - log1p(exp(-diff))
        else
            ΔF[i] += increment - log1p(exp(diff))
        end

        sum_weights[i] += exp(-ΔS + current_bias)
    end

    o.rct += increment + log1p(-1/counter)
    return nothing
end

function ∂V∂Q(o::OPESmultithermal, cv)
    ΔF = o.ΔF
    λ = o.λ
    ΔS_max = maximum(-cv*λ[i] + ΔF[i] for i in eachindex(λ)) # get maximum difference to avoid over/underflow of exp
    denom = 0.0
    der = 0.0

    for i in eachindex(λ)
        diff_i = -cv*λ[i] + ΔF[i]
        add_i = exp(diff_i - ΔS_max)
        denom += add_i
        der += λ[i] * add_i
    end

    return der/denom
end

const opesmt_state_vars = [
    :counter,
    :rct,
    :beta0,
    :λ,
    :ΔF,
]

write_to_file(::OPESmultithermal, ::Nothing) = nothing

function write_to_file(o::OPESmultithermal, filename::String)
    filename=="" && return nothing
    (tmppath, tmpio) = mktemp()
    print(tmpio, rpad("#counter", 25))
    print(tmpio, rpad("rct", 25))
    print(tmpio, rpad("beta_0", 25))
    println(tmpio)
    print(tmpio, rpad(o.counter, 25))
    print(tmpio, rpad(o.rct, 25))
    print(tmpio, rpad(o.beta0, 25))
    println(tmpio, "\n")

    print(tmpio, rpad("#beta_i", 25))
    print(tmpio, rpad("lambda_i", 25))
    print(tmpio, rpad("deltaF_i", 25))
    println(tmpio)

    for i in eachindex(o.β)
        print(tmpio, rpad(o.β[i], 25))
        print(tmpio, rpad(o.λ[i], 25))
        print(tmpio, rpad(o.ΔF[i], 25))
        println(tmpio)
    end

    close(tmpio)
    mv(tmppath, filename; force=true)
    return nothing
end

function opesmt_from_file!(counter, rct, beta, lambda, deltaF, load_bias)
    if load_bias == ""
        return counter, rct, beta, lambda, deltaF
    else
        @level1("|  Getting state from $(load_bias)")
        # state is stored in header, which is always read as a string so we have to parse it
        kernel_data, state_data = readdlm(load_bias; comments=true, header=true)
        counter = parse(Int64, state_data[1])
        rct = parse(Float64, state_data[1])

        beta = kernel_data[:, 1]
        lambda = kernel_data[:, 2]
        deltaF = kernel_data[:, 3]
        return counter, rct, beta, lambda, deltaF
    end
end

function create_buffer(o::OPESmultithermal)
    # for OPES, need to communicate
    # static, counter, sum_weights, sum_weights2, current_bias, (5)
    # current_weight, Z, KDEnorm, old_sum_weights, (4)
    # old_Z, old_KDEnorm, nker, nδker, write_bias_every (5)
    # kernels, δkernels
    # all others are the same between ranks
    return Vector{Float64}(undef, 8+length(o.ΔF))
end

function pack_buffer!(buf, o::OPESmultithermal)
    buf[1] = Float64(o.static)
    buf[2] = Float64(o.counter)
    buf[3] = o.rct
    buf[4] = o.current_bias
    buf[5] = o.current_weight
    buf[6] = o.sum_weights
    buf[7] = o.sum_weights2
    buf[8] = Float64(o.write_bias_every)
    view(buf, 9:length(buf)) .= o.ΔF
    return nothing
end

function unpack_buffer!(o::OPESmultithermal, buf)
    o.static = round(Bool, buf[1])
    o.counter = round(Int64, buf[2])
    o.rct = buf[3]
    o.current_bias = buf[4]
    o.current_weight = buf[5]
    o.sum_weights = buf[6]
    o.sum_weights2 = buf[7]
    o.write_bias_every = round(Int64, buf[8])
    o.ΔF = view(buf, 9:length(buf))
    return nothing
end
