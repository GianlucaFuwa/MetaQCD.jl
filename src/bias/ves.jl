"""
    VES <: AbstractBias

Variationally enhanced sampler that uses a fourier basis to approximate the bias potential.

    VES(p::VESParameters; dummy=false)

Create an instance of a static VES bias using the inputs or the
parameters given in `p`.

# Specifiable parameters
`kind_of_cv::String = "topcharge_clover"` - Collective variable
`numsmears_for_cv::Int64 = 4` - Number of smearing steps for the CV (step size is given in superstructure `Bias`)
`cvlims::NTuple{2, Float64} = (-6, 6)` - Minimum and maximum of the explorable cv-space;
must be ordered \\
`penalty_weight::Float64 = 1000` - Penalty when cv is outside of `cvlims`; must be positive \\
`nbasis::Int64 = 20` - Number of basis functions used in approximation \\
`batch_size::Int64 = 50` - Number CV measurements before optimization iteration \\
"""
mutable struct VES{CV} <: AbstractBias
    cvinfo::CV
    static::Bool
    cvlims::NTuple{2,Float64}
    penalty_weight::Float64
    step_size::Float64
    nbasis::Int64
    batch_size::Int64    
    n::Int64

    alpha::Vector{Float64} # instantaneous ptimization parameters
    alpha_bar::Vector{Float64} # averaged optimization parameters
    s_vec::Vector{Float64}
    w_sum::Float64

    gradient::Vector{Float64}
    hessian::Matrix{Float64}
    # temp::Vector{Float64} # temp for matrix vector product
    write_bias_every::Int64
end

function VES(p::VESParameters; instance=MPI_INSTANCE[], dummy=false, build=false)
    inum = if dummy
        0
    else
        instance
    end
    cvinfo = get_cvinfo_from_parameters(p)
    static = p.static
    cvlims = !dummy ? tuple(p.cvlims...) : (-Inf, Inf)
    @level1("|  CVLIMS: $(string(cvlims))")
    @assert cvlims[1] == -cvlims[2] "CV lims must be symmetric for VES"
    penalty_weight = !dummy ? p.penalty_weight : 0.0
    @level1("|  PENALTY WEIGHT: $(penalty_weight)")
    step_size = p.step_size
    @level1("|  STEP SIZE: $(step_size)")
    nbasis = p.nbasis
    @level1("|  NBASIS: $(nbasis)")
    batch_size = p.batch_size
    @level1("|  BATCH SIZE: $(batch_size)")
    write_bias_every = batch_size
    @level1("|  WRITE BIAS EVERY: $(write_bias_every)")

    alpha = isempty(p.alpha) ? zeros(nbasis+1) : p.alpha
    alpha_bar = deepcopy(alpha)
    s_vec = zeros(0)
    n = 0
    w_sum = 0.0
    
    gradient = zeros(nbasis+1)
    hessian = zeros(nbasis+1, nbasis+1)
    return VES(
        cvinfo, static, cvlims, penalty_weight, step_size, nbasis, batch_size, n,
        alpha, alpha_bar, s_vec, w_sum,
        gradient, hessian, write_bias_every
    )
end

get_ext(::VES) = ".ves"
is_adaptive(::VES) = false
set_sigma0!(::VES, ::Any) = nothing
clear!(::VES) = nothing

function update!(p::VES, cv, itrj)
    if !p.static
        push!(p.s_vec, cv...)

        if itrj % p.batch_size == 0
            n = p.n
            α = p.alpha
            ᾱ = p.alpha_bar
            calc_gradient_and_hessian!(p)
            α .-= p.step_size * (p.gradient + p.hessian * (α - ᾱ))
            w_new = log(n + 1 + 1e-9)
            @. ᾱ = (p.w_sum * ᾱ + w_new * α) / (p.w_sum + w_new)
            p.n += 1
            p.w_sum += w_new
            empty!(p.s_vec)
        end
    end

    return nothing
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
    ᾱ = p.alpha
    lim = abs(p.cvlims[1])

    out = ᾱ[1]

    for i in 1:nbasis
        out += ᾱ[i+1] * cos(i*π*cv / lim)
    end

    return out
end

function ∂V∂Q(p::VES, cv)
    nbasis = p.nbasis
    ᾱ = p.alpha
    lim = abs(p.cvlims[1])
    out = 0.0

    for i in 1:nbasis
        out += -i*π/lim * ᾱ[i+1] * sin(i*π*cv / lim)
    end

    return out
end

function calc_gradient_and_hessian!(p::VES)
    nbasis = p.nbasis
    gradient = p.gradient
    hessian = p.hessian
    s_vec = p.s_vec
    lim = abs(p.cvlims[1])
    n = length(s_vec)

    # Calculate gradient
    gradient[1] = 0.0

    for i in 1:nbasis
        sum_cos = 0.0
        for k in 1:n
            s = s_vec[k]
            sum_cos += cos(i*π*s / lim) # we only need the cos terms because we force the potential to be symmetric, i.e., V(-s) = V(s)
        end
        gradient[i+1] = -sum_cos / n
    end

    # Calculate hessian
    hessian[1, 1] = 0.0
    hessian[1, 2] = 0.0
    hessian[2, 1] = 0.0

    for i in 1:nbasis
        mean_cos_i = -gradient[i+1]
        
        for j in 1:nbasis
            mean_cos_j = -gradient[j+1]
            
            cov_cc = 0.0
            
            for k in 1:n
                s = s_vec[k]
                cos_i = cos(i*π*s / lim)
                cos_j = cos(j*π*s / lim)
                
                diff_cos_i = cos_i - mean_cos_i
                diff_cos_j = cos_j - mean_cos_j
                
                cov_cc += diff_cos_i * diff_cos_j
            end
            
            hessian[i+1, j+1] = cov_cc / (n - 1)
        end
    end

    return nothing
end

write_to_file(::VES, ::Nothing, args...) = nothing

function write_to_file(p::VES, filename::AbstractString, clear=false)
    filename == "" && return nothing
    mode = clear ? "w" : "a"
    set_ext!(filename, MPI_INSTANCE[], Val(3))
    open(filename, mode) do io
        for i in 1:p.nbasis+1
            print(io, "$(p.alpha[i])\t")
        end
        for i in 1:p.nbasis+1
            print(io, "$(p.alpha_bar[i])\t")
        end

        println(io)
    end
    # (tmppath, tmpio) = mktemp() # open temporary file at arbitrary location in storage
    # println(tmpio, "$(rpad("alpha", 25))\t$(rpad("alpha_bar", 7))")
    #
    # for i in 1:p.nbasis+1
    #     println(tmpio, "$(rpad(p.alpha[i], 25))\t$(rpad(p.alpha_bar[i], 25))")
    # end
    #
    # close(tmpio)
    # mv(tmppath, filename; force=true) # replace bias file with temporary file
    return nothing
end

function create_buffer(p::VES)
    # for VES, only need to communicate static, write_bias_every, batch_size, nbasis, step_size and values
    # all others are the same between ranks
    buf = Vector{Float64}(undef, 5+2length(p.alpha))
    buf[1] = Float64(p.static)
    buf[2] = Float64(p.write_bias_every)
    buf[3] = Float64(p.batch_size)
    buf[4] = Float64(p.nbasis)
    buf[5] = Float64(p.step_size)
    buf[6:5+length(p.alpha)] .= p.alpha
    buf[6+length(p.alpha):end] .= p.alpha_bar
    return buf
end

function unpack_buffer!(p::VES, buf)
    len_alpha = div(length(buf)-5, 2)
    p.static = round(Bool, buf[1])
    p.write_bias_every = round(Int64, buf[2])
    p.batch_size = round(Int64, buf[3])
    p.nbasis = round(Int64, buf[4])
    p.step_size = round(Float64, buf[5])
    p.alpha = buf[6:5+len_alpha]
    p.alpha_bar = buf[6+len_alpha:length(buf)]
    return nothing
end
