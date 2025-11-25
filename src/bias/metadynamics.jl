"""
    Metadynamics <: AbstractBias

Metadynamics bias-enhanced sampler from https://arxiv.org/abs/cond-mat/0208352 .

    Metadynamics(p::MetadynamicsParameters; dummy=false)

Create an instance of a Metadynamics bias using the inputs or the parameters given in `p`.

# Specifiable parameters
`kind_of_cv::String = "topcharge_clover"` - Collective variable
`numsmears_for_cv::Int64 = 4` - Number of smearing steps for the CV (step size is given in superstructure `Bias`)
`symmetric::Bool = true` - If `true`, the bias is built symmetrically by updating for both cv and
-cv at every update-iteration \\
`stride::Int64 = 1` - Number of iterations between updates; must be >0 \\
`cvlims::NTuple{2, Float64} = (-6, 6)` - Minimum and maximum of the explorable cv-space;
must be ordered \\
`write_bias_every::Int64 = 1` - Number of update iterations between writes of the bias to file \\
`biasfactor::Float64 = Inf` - Biasfactor for well-tempered Metadynamics; must be >1 \\
`bin_width::Float64 = 0.1` - Width of bins in histogram; must be >0 \\
`weight::Float64 = 0.01` - (Starting) Height of added Gaussians; must be positive \\
`penalty_weight::Float64 = 1000` - Penalty when cv is outside of `cvlims`; must be positive \\
"""
mutable struct Metadynamics{CV} <: AbstractBias
    cvinfo::CV
    static::Bool
    symmetric::Bool
    stride::Int64
    cvlims::NTuple{2,Float64}

    biasfactor::Float64
    bin_width::Float64
    weight::Float64
    penalty_weight::Float64

    bin_vals::Vector{Float64}
    values::Vector{Float64}
    write_bias_every::Int64
end

function Metadynamics(
    p::MetadynamicsParameters; instance=1, dummy=false, mpi_multi_sim=false, build=false
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
    symmetric = p.symmetric
    stride = p.stride
    @level1("|  STATIC: $(static)")
    @level1("|  STRIDE: $(stride)")
    @assert stride > 0 "STRIDE must be >0"

    @level1("|  CVLIMS: $(string(p.cvlims))")
    @assert issorted(p.cvlims) "CVLIMS must be sorted from low to high"

    @level1("|  BIN_WIDTH: $(p.bin_width)")
    @assert p.bin_width > 0 "BIN_WIDTH must be > 0"

    if (0 < instance <= length(p.load_bias) && !dummy)
        bin_vals, values = metad_from_file(p, p.load_bias[instance+1])
    elseif build && (length(p.load_bias) != 0)
        bin_vals, values = metad_from_file(p, p.load_bias[1])
    else
        bin_vals, values = metad_from_file(p, "")
    end

    @level1("|  META_WEIGHT: $(p.weight)")
    @assert p.weight > 0 "METAD.WEIGHT must be > 0"

    @level1("|  PENALTY_WEIGHT: $(p.penalty_weight)")

    biasfactor = p.biasfactor
    @level1("|  BIASFACTOR: $(biasfactor)")
    @assert biasfactor > 1 "BIASFACTOR must be > 1"

    write_bias_every = if p.write_bias_every <= stride
        stride
    else
        p.write_bias_every
    end
    @level1("|  WRITE_BIAS_EVERY: $(string(dummy ? "" : write_bias_every))")
    @level1("|")
    return Metadynamics(
        cvinfo,
        static,
        symmetric,
        stride,
        tuple(p.cvlims...),
        biasfactor,
        p.bin_width,
        p.weight,
        p.penalty_weight,
        bin_vals,
        values,
        write_bias_every,
    )
end

Base.length(m::Metadynamics) = length(m.values)
Base.eachindex(m::Metadynamics) = eachindex(m.values)
Base.lastindex(m::Metadynamics) = lastindex(m.values)
get_ext(::Metadynamics) = ".metad"
is_adaptive(::Metadynamics) = false
set_sigma0!(::Metadynamics, ::Any) = nothing
ext_length(::Metadynamics) = Val(5)

function Base.setindex!(m::Metadynamics, v, i)
    return m.values[i] = v
end

@inline function Base.getindex(m::Metadynamics, i)
    return m.values[i]
end

@inline function index(m::Metadynamics, cv)
    idx = (cv - m.cvlims[1]) / m.bin_width + 0.5
    return round(Int64, idx, RoundNearestTiesAway)
end

function update!(m::Metadynamics, cv, args...)
    cv_range = if length(cv) > 1
        range(1, length(cv); step=cld(length(cv), 20))
    else
        range(1, length(cv))
    end

    for i in cv_range
        cvᵢ = cv[i]
        for (idx, bin_val) in enumerate(m.bin_vals)
            wt = exp(-m[idx] / m.biasfactor)
            m[idx] += m.weight * wt * exp(-0.5(cvᵢ - bin_val)^2 / m.bin_width^2)
        end

        if m.symmetric
            for (idx, bin_val) in enumerate(m.bin_vals)
                wt = exp(-m[idx] / m.biasfactor)
                m[idx] += m.weight * wt * exp(-0.5(-cvᵢ - bin_val)^2 / m.bin_width^2)
            end
        end
    end

    return nothing
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

function clear!(m::Metadynamics)
    @batch for i in eachindex(m)
        m[i] = 0.0
    end

    return nothing
end

write_to_file(::Metadynamics, ::Nothing, args...) = nothing

function write_to_file(m::Metadynamics, filename::AbstractString, args...)
    filename == "" && return nothing
    (tmppath, tmpio) = mktemp() # open temporary file at arbitrary location in storage
    println(tmpio, "$(rpad("CV", 7))\t$(rpad("V(CV)", 7))")

    for i in eachindex(m)
        println(tmpio, "$(rpad(m.bin_vals[i], 7, "0"))\t$(rpad(m.values[i], 7, "0"))")
    end

    close(tmpio)
    mv(tmppath, filename; force=true) # replace bias file with temporary file
    return nothing
end

function metad_from_file(p, filename)
    cvlims = p.cvlims

    if filename == ""
        bin_vals = range(cvlims[1], cvlims[2]; step=p.bin_width)
        values = zero(bin_vals)
        @level1("|  initialized as zeros")
        return collect(bin_vals), values
    else
        values, _ = readdlm(filename, Float64; header=true)
        bin_vals = range(cvlims[1], cvlims[2]; step=p.bin_width)
        @assert length(values[:, 2]) == length(bin_vals) "your bias doesn't match parameters"
        @level1("|  initialized from \"$(filename)\"")
        return collect(bin_vals), values[:, 2]
    end
end

function create_buffer(m::Metadynamics)
    # for Metadynamics, only need to communicate static, write_bias_every and values
    # all others are the same between ranks
    return Vector{Float64}(undef, 2+length(m.values))
end

function pack_buffer!(buf, m::Metadynamics)
    buf[1] = Float64(m.static)
    buf[2] = Float64(m.write_bias_every)
    buf[3:end] .= m.values
end

function unpack_buffer!(m::Metadynamics, buf)
    m.static = round(Bool, buf[1])
    m.write_bias_every = round(Int64, buf[2])
    m.values .= view(buf, 3:length(buf))
end
