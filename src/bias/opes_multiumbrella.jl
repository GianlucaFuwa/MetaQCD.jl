"""
    OPESmultiumbrella{CV} <: AbstractBias
OPES MultiUmbrella from https://arxiv.org/abs/2007.03055 (Sec. VI).
    OPESmultiumbrella(; stride=1, cv_min_max=(0.0, 0.2), cv_num=20, σ=0.02, barrier=Inf)
Create an instance of an OPES multiumbrella bias, targeting a uniform coverage of the
collective variable `cv` (e.g. |Polyakov loop|) over the range `cv_min_max`.
# Specifiable parameters
`stride::Int64 = 1` - Number of iterations between updates; must be >0 \\
`write_bias_every::Int64 = 1` - Number of update iterations between writes of the bias to file \\
`cv_min_max::Vector{Float64} = [0.0, 0.2]` - Minimum and maximum of the CV range (must be ordered) \\
`cv_num::Int64 = 20` - Number of umbrella centers between `cv_min` and `cv_max` \\
`σ::Float64 = 0.02` - Width of each Gaussian umbrella; should roughly match the unbiased
fluctuation width of the CV, or the smallest feature you want the FES to resolve \\
`barrier::Float64 = Inf` - Optional cap on the initial ΔF estimate (Appendix B of the paper),
useful only if the very first biased steps are unstable due to a too-strong initial bias \\
"""
mutable struct OPESmultiumbrella{CV} <: AbstractBias
    cvinfo::CV
    static::Bool
    is_first_step::Bool
    stride::Int64
    counter::Int64
    rct::Float64
    current_bias::Float64
    current_weight::Vector{Float64}
    σ::Float64
    cv_min::Float64
    cv_max::Float64
    sλ::Vector{Float64}
    ΔF::Vector{Float64}
    sum_weights::Vector{Float64}
    sum_weights2::Vector{Float64}
    barrier::Float64
    write_bias_every::Int64
end

function OPESmultiumbrella(
    p::OPESmultiumbrellaParameters;
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
    @assert length(p.cv_min_max) == 2 "cv_min_max has to be a vector of length 2"
    cv_min = minimum(p.cv_min_max)
    cv_max = maximum(p.cv_min_max)
    @assert p.cv_num > 0 "cv_num has to be bigger than 0"
    @assert p.sigma > 0 "σ has to be bigger than 0"
    sλ = collect(range(cv_min, cv_max, p.cv_num))
    ΔF = zero(sλ)
    sum_weights = zero(sλ)
    sum_weights2 = zero(sλ)
    current_weight = zero(sλ)
    barrier = dummy ? Inf : p.barrier

    if build && (length(p.load_bias) != 0)
        load_bias = p.load_bias[1]
        is_first_step = false
        counter, rct, sλ, ΔF = opesmu_from_file!(counter, rct, sλ, ΔF, load_bias)
    elseif (0 < instance <= length(p.load_bias) && !dummy)
        load_bias = p.load_bias[instance+1]
        is_first_step = false
        counter, rct, sλ, ΔF = opesmu_from_file!(counter, rct, sλ, ΔF, load_bias)
    end

    write_bias_every = if p.write_bias_every <= stride
        stride
    else
        p.write_bias_every
    end
    @level1("|  STATIC: $(static)")
    @level1("|  STRIDE: $(stride)")
    @assert stride > 0 "STRIDE must be >0"
    @level1("|  CV MIN: $(cv_min)")
    @level1("|  CV MAX: $(cv_max)")
    @level1("|  NUM UMBRELLAS: $(p.cv_num)")
    @level1("|  SIGMA: $(p.sigma)")
    @level1("|  WRITE_BIAS_EVERY: $(string(dummy ? "" : write_bias_every))")
    return OPESmultiumbrella(
        cvinfo, static, is_first_step, stride, counter, rct, 0.0, current_weight,
        p.sigma, cv_min, cv_max, sλ, ΔF, sum_weights, sum_weights2, barrier, write_bias_every
    )
end

get_ext(::OPESmultiumbrella) = ".opesmu"
is_adaptive(o::OPESmultiumbrella) = false
set_sigma0!(::OPESmultiumbrella, ::Any) = nothing
ext_length(::OPESmultiumbrella) = Val(6)

@inline Δu(o::OPESmultiumbrella, cv, i) = (cv - o.sλ[i])^2 / (2*o.σ^2)

function (o::OPESmultiumbrella)(cv)
    calculate!(o, cv)
    return o.current_bias
end

function calculate!(o::OPESmultiumbrella, cv::Float64)
    ΔF = o.ΔF
    sλ = o.sλ
    ΔS_max = maximum(-Δu(o, cv, i) + ΔF[i] for i in eachindex(sλ))
    sum = 0.0
    for i in eachindex(sλ)
        diff_i = -Δu(o, cv, i) + ΔF[i]
        sum += exp(diff_i - ΔS_max)
    end
    for i in eachindex(sλ)
        o.current_weight[i] = exp(-ΔS_max) - sum/length(sλ)
    end
    current_bias = -(ΔS_max + log(sum/length(sλ)))
    o.current_bias = current_bias
    return nothing
end

function update!(o::OPESmultiumbrella, cv, itrj)
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

function update_ΔF!(o::OPESmultiumbrella, current_bias::Float64, cv::Float64)
    o.counter += 1
    counter = o.counter
    rct = o.rct
    arg = current_bias - rct - log(counter-1)
    sλ = o.sλ
    ΔF = o.ΔF
    sum_weights = o.sum_weights
    increment = arg > 0 ? arg + log1p(exp(-arg)) : log1p(exp(arg))
    for i in eachindex(sλ)
        ΔS = Δu(o, cv, i)
        diff = -ΔS + current_bias - rct + ΔF[i] - log1p(counter - 1)
        if o.is_first_step === false && counter == 2 && isfinite(o.barrier)
            # optional: cap the very first ΔF estimate (Appendix B) to avoid an
            # overly strong initial push toward far-away umbrella centers
            ΔF[i] = min(ΔF[i], o.barrier)
        end
        if diff > 0
            ΔF[i] += increment - diff - log1p(exp(-diff))
        else
            ΔF[i] += increment - log1p(exp(diff))
        end
        sum_weights[i] += exp(-ΔS + current_bias)
    end
    o.rct += increment + log1p(-1/counter)
    return nothing
end

"""
    ∂V∂Q(o::OPESmultiumbrella, cv)
Returns dv/d(cv), i.e. ∂v/∂|L| for the multiumbrella bias, to be chained with
∂|L|/∂U (e.g. your existing `polyakov_mag_deriv!`) to get the force.
"""
function ∂V∂Q(o::OPESmultiumbrella, cv)
    ΔF = o.ΔF
    sλ = o.sλ
    σ2 = o.σ^2
    ΔS_max = maximum(-Δu(o, cv, i) + ΔF[i] for i in eachindex(sλ))
    denom = 0.0
    der = 0.0
    for i in eachindex(sλ)
        diff_i = -Δu(o, cv, i) + ΔF[i]
        add_i = exp(diff_i - ΔS_max)
        denom += add_i
        der += ((cv - sλ[i]) / σ2) * add_i
    end
    return der/denom
end

const opesmu_state_vars = [
    :counter,
    :rct,
    :σ,
    :sλ,
    :ΔF,
]

write_to_file(::OPESmultiumbrella, ::Nothing, args...) = nothing
function write_to_file(o::OPESmultiumbrella, filename::AbstractString, args...)
    filename=="" && return nothing
    (tmppath, tmpio) = mktemp()
    print(tmpio, rpad("#counter", 25))
    print(tmpio, rpad("rct", 25))
    print(tmpio, rpad("sigma", 25))
    println(tmpio)
    print(tmpio, rpad(o.counter, 25))
    print(tmpio, rpad(o.rct, 25))
    print(tmpio, rpad(o.σ, 25))
    println(tmpio, "\n")
    print(tmpio, rpad("#s_lambda_i", 25))
    print(tmpio, rpad("deltaF_i", 25))
    println(tmpio)
    for i in eachindex(o.sλ)
        print(tmpio, rpad(o.sλ[i], 25))
        print(tmpio, rpad(o.ΔF[i], 25))
        println(tmpio)
    end
    close(tmpio)
    mv(tmppath, filename; force=true)
    return nothing
end

function opesmu_from_file!(counter, rct, sλ, deltaF, load_bias)
    if load_bias == ""
        return counter, rct, sλ, deltaF
    else
        @level1("|  Getting state from $(load_bias)")
        kernel_data, state_data = readdlm(load_bias; comments=true, header=true)
        counter = parse(Int64, state_data[1])
        rct = parse(Float64, state_data[1])
        sλ = kernel_data[:, 1]
        deltaF = kernel_data[:, 2]
        return counter, rct, sλ, deltaF
    end
end

function create_buffer(o::OPESmultiumbrella)
    return Vector{Float64}(undef, 8+length(o.ΔF))
end

function pack_buffer!(buf, o::OPESmultiumbrella)
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

function unpack_buffer!(o::OPESmultiumbrella, buf)
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
