"""
    MetaBias(
        ; ensemblename::String="",
        filname = nothing,
        which = nothing,
        stream::Int = 0,
    )

Create a `MetaBias` object using the bias from stream `stream` in the directory `ensemblename`.
or directly from the file `filename`.
This serves as a functor returning the bias value at an input cv. \\
Make sure the directory only contains bias files produced by MetaQCD.jl or in
the same format. If the file extension is not .metad or .opes then you will need to
specify `which` as either `:metad` or `:opes`.
"""
struct MetaBias{F}
    bias::F
    ensemblename::String
    ext::String
end

function MetaBias(
    ; filename=nothing, ensemblename::String="", which=nothing, stream=0
)
    from_ensemble = ensemblename != "" && isnothing(filename)
    dir = if isabspath(ensemblename) && from_ensemble
        joinpath(ensemblename, "biaspotentials/")
    elseif from_ensemble
        path = joinpath(splitpath(@__DIR__())[1:end-2]...) * "/ensembles/$(ensemblename)/biaspotentials/"
        @assert ispath(path) """
        Ensemble \"$(ensemblename)\" could not be found or doesn't exist.
        """
        path
    else
        ""
    end

    file, ext = if from_ensemble
        @assert isdir(dir) "Directory \"$(dir)\" doesn't exist."
        filenames = readdir(dir)
        fname = try
            filenames[findfirst(x -> occursin("00$(stream)", x), filenames)]
        catch _
            filenames[findfirst(x -> occursin("00$(stream+1)", x), filenames)]
        end

        _file = dir * fname
        _ext = splitext(_file)[end]
        _file, _ext
    else
        filename, splitext(filename)[end]
    end

    if ext == ".metad" || which == :metad
        data = readdlm(file; skipstart=1)
        cvlims = data[1, 1], data[end, 1]
        bin_width = data[2, 1] - data[1, 1]
        bin_vals = data[:, 1]
        values = data[:, end]
        bias = Metadynamics(
            true,
            1,
            cvlims,
            Inf,
            bin_width,
            1.0,
            100,
            bin_vals,
            values,
        )
    elseif ext == ".opes" || which == :opes
        bias = OPES(file)
    elseif ext == ".ves" || which == :ves
    elseif ext == ".opesmt" || which == :opesmt
        bias = OPESmultithermal(file)
    else
        throw(AssertionError("File extension $ext not recognized.
                             Must be either .metad or .opes"))
    end

    ename = split(ensemblename, "/")[end]
    mbias = MetaBias(bias, string(ename), ext)
    plt = plot(mbias)
    display(plt)
    return mbias, plt
end

(m::MetaBias{F})(cv::Float64) where {F} = m.bias(cv)
Base.nameof(::MetaBias{F}) where {F} = nameof(F)

function Base.show(io::IO, m::MetaBias)
    print(io, "MetaBias{$(typeof(m.bias))}(ensemble: \"$(m.ensemblename)\")")
    return nothing
end

RecipesBase.@recipe function f(
    b::MetaBias; cvlims=nothing, normalize=false, ylims=nothing
)
    bias = b.bias
    xlims = if b.bias isa OPESmultithermal
        cvlims = extrema(b.bias.β)
    else
        cvlims ≡ nothing ? bias.cvlims : cvlims
    end
    yylims = ylims ≡ nothing ? :auto : ylims
    isinf(sum(xlims)) && (xlims = (-6, 6))

    legend := false
    xticks --> floor(xlims[1]):ceil(xlims[2])
    xlims := (xlims[1], xlims[2])
    ylims := yylims
    xlabel --> "Collective Variable"
    ylabel --> "Bias Potential ($(nameof(b)))"
    title --> b.ensemblename
    titlefontsize --> 10
    x = if bias isa OPESmultithermal
        xlims[1]:0.001:xlims[2]
    else
        (bias isa OPES) ? (xlims[1]:0.001:xlims[2]-0.001) : bias.bin_vals
    end
    yraw = b.(x)
    y = normalize ? yraw .- maximum(yraw) : yraw
    return x, y
end

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
        idx == length(m.values) && return m[idx]
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

mutable struct OPES
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
end

function OPES(filename::String)
    @assert isfile(filename) "file \"$(filename)\" doesn't exist"
    state = Dict{Symbol,Any}()
    kernels, nker = opes_from_file!(state, filename)
    is_first_step = false
    counter = Int64(state[:counter])
    biasfactor = state[:biasfactor]
    sigma0 = state[:sigma0]
    epsilon = state[:epsilon]
    sum_weights = state[:sum_weights]
    sum_weights2 = state[:sum_weights2]
    KDEnorm = state[:KDEnorm]
    explore = KDEnorm == counter ? true : false
    bias_prefactor = explore ? (biasfactor - 1) : (1 - 1 / biasfactor)
    Z = state[:Z]
    threshold = state[:threshold]
    cutoff2 = state[:cutoff2]
    penalty = state[:penalty]

    return OPES(
        is_first_step,
        explore, true, counter, 1, (-3, 3),
        biasfactor, bias_prefactor,
        sigma0, 1e-6, false,
        epsilon, sum_weights, sum_weights2, 0.0, 0.0, false, Z, KDEnorm,
        threshold, cutoff2, penalty,
        sum_weights, Z, sum_weights,
        nker, kernels, 0, Vector{Kernel}(undef, 2),
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
    cutoff² = o.cutoff2
    penalty = o.penalty

    prob = 0.0

    for kernel in o.kernels
        prob += kernel(cv, cutoff², penalty)
        @assert prob < 1e10 "opes_prob = $prob is too high, something probably went wrong"
    end

    prob /= o.KDEnorm
    current_bias = o.bias_prefactor * log(prob / o.Z + o.epsilon)
    o.current_weight = prob
    o.current_bias = current_bias
    return nothing
end

function ∂V∂Q(o::OPES, cv)
    cutoff² = o.cutoff2
    penalty = o.penalty
    prob = 0.0
    deriv = 0.0

    for kernel in o.kernels
        prob += kernel(cv, cutoff², penalty)
        deriv += derivative(kernel, cv, cutoff², penalty)
    end

    prob /= o.KDEnorm
    deriv /= o.KDEnorm
    Z = o.Z
    out = -o.bias_prefactor / (prob / Z + o.epsilon) * deriv / Z
    return out
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

function opes_from_file!(dict, usebias)
    if usebias == ""
        kernels = Vector{Kernel}(undef, 0)
        return kernels, 0
    else
        # state is stored in header, which is always read as a string so we have to parse it
        kernel_data, state_data = readdlm(usebias; comments=true, header=true)
        state_parse = [parse(Float64, state_data[i]) for i in eachindex(state_data)]
        len_vars = length(opes_state_vars)
        @assert length(state_parse) ∈ (len_vars, len_vars-1)

        if length(state_parse) == len_vars
            for (i, state_var) in enumerate(opes_state_vars)
                dict[state_var] = state_parse[i]
            end
        else # INFO: For old format (pre v1.1.1)
            j = 1

            for state_var in opes_state_vars
                if state_var === :KDEnorm
                    dict[:KDEnorm] = dict[:sum_weights]
                elseif state_var === :cutoff2
                    dict[:cutoff2] = state_parse[j]^2
                    j += 1
                else
                    dict[state_var] = state_parse[j]
                    j += 1
                end
            end
        end

        kernels = Vector{Kernel}(undef, size(kernel_data, 1))

        for i in axes(kernel_data, 1)
            kernels[i] = Kernel(view(kernel_data, i, 1:3)...)
        end

        return kernels, length(kernels)
    end
end

mutable struct VES
    cvlims::NTuple{2,Float64}
    penalty_weight::Float64
    nbasis::Int64
    alpha::Vector{Float64} # instantaneous ptimization parameters
    alpha_bar::Vector{Float64} # averaged optimization parameters
end

function VES(filename::String)
    @assert isfile(filename) "file \"$(filename)\" doesn't exist"
    data = readdlm(filename)
    nbasis = size(data, 2) ÷ 2 - 2
    alpha = data[end, 1:nbasis+2]
    alpha_bar = data[end, nbasis+3:end]
    penalty_weight = 100
    cvlims = (-3, 3)
    return VES(cvlims, penalty_weight, nbasis, alpha, alpha_bar)
end

function (p::VES)(cv)
    lb, ub = p.cvlims

    out = if !in_bounds(cv, p.cvlims...)
        bounds_penalty = 100
        which_bound, dist² = findmin(((cv - lb)^2, (cv - ub)^2))
        nearest_bound = which_bound == 1 ? lb : ub
        return_potential(p, nearest_bound) + bounds_penalty * dist²
    else
        return_potential(p, cv)
    end

    return out
end

function return_potential(p::VES, cv)
    nbasis = p.nbasis
    ᾱ = p.alpha_bar
    lim = abs(p.cvlims[1])
    out = ᾱ[1] + ᾱ[2] * cv^2

    for i in 1:nbasis
        out += ᾱ[i+2] * cos(i*π*cv / lim)
    end

    return out
end

mutable struct OPESmultithermal
    counter::Int64
    rct::Float64
    β::Vector{Float64}
    λ::Vector{Float64}
    ΔF::Vector{Float64}
end

function OPESmultithermal(filename::String)
    @assert isfile(filename) "file \"$(filename)\" doesn't exist"
    # state is stored in header, which is always read as a string so we have to parse it
    kernel_data, state_data = readdlm(filename; comments=true, header=true)
    counter = parse(Int64, state_data[1])
    rct = parse(Float64, state_data[1])

    beta = kernel_data[:, 1]
    lambda = kernel_data[:, 2]
    deltaF = kernel_data[:, 3]
    return OPESmultithermal(
        counter, rct, beta, lambda, deltaF
    )
end

function (o::OPESmultithermal)(cv)
    return calculate!(o, cv)
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

    current_bias = -(ΔS_max + log(sum/length(λ)))
    return current_bias
end

function bias_from_weights(
    meas::MetaMeasurements, cvlims=(-3, 3); bin_width=0.01, kernel_variance=0.0
)
    observables = meas.observables
    numinstances = count(x->contains(string(x), "bias_data"), observables)
    @assert numinstances > 0 "No bias data found"
    @assert kernel_variance >= 0.0 "kernel_variance has to be >= 0.0"

    bias_data = [getproperty(meas, Symbol("bias_data_00", i)) for i in 0:numinstances-1]
    q = range(cvlims[1], cvlims[2]; step=bin_width)
    V = zero(q)

    if kernel_variance == 0.0
        for i in 1:numinstances
            for (j, cv) in enumerate(bias_data[i]["cv1"])
                val, idx = findmin(x->abs(cv-x), q)
                val, idxn = findmin(x->abs(-cv-x), q)
                V[idx] += bias_data[i]["weight_tiwari"][j] # TODO: generalize for diff weights
                V[idxn] += bias_data[i]["weight_tiwari"][j]
            end
        end
    else
        for i in 1:numinstances
            for (j, cv) in enumerate(bias_data[i]["cv1"])
                w = bias_data[i]["weight_tiwari"][j] # TODO: generalize for diff weights
                for (idx, bin_val) in enumerate(q)
                    V[idx] += w * exp(-0.5(cv - bin_val)^2 / kernel_variance^2)
                end

                for (idx, bin_val) in enumerate(q)
                    V[idx] += w * exp(-0.5(-cv - bin_val)^2 / kernel_variance^2)
                end
            end
        end
    end

    V ./= sum(V)
    return q, log.(V)
end
