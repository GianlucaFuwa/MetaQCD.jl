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
    p::MetadynamicsParameters;
    instance=MPI_INSTANCE[], dummy=false, mpi_multi_sim=false, build=false
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
        p.static
    end
    symmetric = p.symmetric
    stride = p.stride
    @level1("|    STATIC: $(static)")
    @level1("|    STRIDE: $(stride)")
    @assert stride > 0 "STRIDE must be >0"
    cvlims = p.cvlims

    if build && (length(p.load_bias) != 0)
        bin_width, bin_vals, values = metad_from_file(p, p.load_bias[1])
        if extrema(bin_vals) != tuple(p.cvlims...)
            @level1("|    @Info: cvlims of `load_bias` were different than provided `cvlims` in parameter file")
            cvlims = extrema(bin_vals)
        end
    elseif (0 <= instance <= length(p.load_bias)-1 && !dummy)
        bin_width, bin_vals, values = metad_from_file(p, p.load_bias[inum+1])
        if extrema(bin_vals) != tuple(p.cvlims...)
            @level1("|    @Info: cvlims of `load_bias` were different than provided `cvlims` in parameter file")
            cvlims = extrema(bin_vals)
        end
    else
        bin_width, bin_vals, values = metad_from_file(p, "")
    end

    @level1("|    CVLIMS: $(string(cvlims))")
    @assert issorted(p.cvlims) "CVLIMS must be sorted from low to high"

    @level1("|    BIN_WIDTH: $(bin_width)")
    @assert bin_width > 0 "BIN_WIDTH must be > 0"
    @level1("|    META_WEIGHT: $(p.weight)")
    @assert p.weight > 0 "METAD.WEIGHT must be > 0"

    @level1("|    PENALTY_WEIGHT: $(p.penalty_weight)")

    biasfactor = p.biasfactor
    @level1("|    BIASFACTOR: $(biasfactor)")
    @assert biasfactor > 1 "BIASFACTOR must be > 1"

    write_bias_every = if p.write_bias_every <= stride
        stride
    else
        p.write_bias_every
    end
    @level1("|    WRITE_BIAS_EVERY: $(string(dummy ? "" : write_bias_every))")
    return Metadynamics(
        cvinfo,
        static,
        symmetric,
        stride,
        tuple(cvlims...),
        biasfactor,
        bin_width,
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
        range(1, length(cv); step=cld(length(cv), 40))
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
    set_ext!(filename, MPI_INSTANCE[], Val(5))
    (tmppath, tmpio) = mktemp() # open temporary file at arbitrary location in storage

    if isfile(filename)
        old_values, old_header = readdlm(filename; header=true)
        new_header = hcat(old_header, "$(rpad("V(CV)", 7))")
        new_values = hcat(old_values, m.values)
    else
        new_header = ["$(rpad("CV", 7))" "$(rpad("V(CV)", 7))"]
        new_values = hcat(m.bin_vals, m.values)
    end

    final_output = vcat(new_header, new_values)
    writedlm(tmpio, final_output, '\t')
    close(tmpio)
    
    mv(tmppath, filename; force=true) # replace bias file with temporary file
    return nothing
end

function metad_from_file(p, filename)
    cvlims = p.cvlims

    if filename == ""
        bin_width = p.bin_width
        bin_vals = range(cvlims[1], cvlims[2]; step=bin_width)
        values = zero(bin_vals)
        @level1("|  initialized as zeros")
        return bin_width, collect(bin_vals), values
    else
        values, _ = readdlm(filename, Float64; header=true)
        @assert length(values[:, 1]) == length(values[:, end]) """
        the number of bin edges and the number of values isn't the same in your provided
        bias file
        """
        bin_width = abs(round(values[1, 1] - values[2, 1]; sigdigits=5))
        @level1("|  initialized from \"$(filename)\"")
        return bin_width, values[:, 1], values[:, end]
    end
end

function create_buffer(m::Metadynamics)
    # for Metadynamics, need to communicate static, write_bias_every, bin_width, bin_vals and values
    buf = Vector{Float64}(undef, 5+2length(m.values))
    buf[1] = Float64(m.static)
    buf[2] = Float64(m.write_bias_every)
    buf[3] = Float64(m.bin_width)
    buf[4] = Float64(m.cvlims[1])
    buf[5] = Float64(m.cvlims[2])
    buf[6:5+length(m.bin_vals)] .= m.bin_vals
    buf[6+length(m.bin_vals):end] .= m.values
    return buf
end

function unpack_buffer!(m::Metadynamics, buf)
    len_vals = div(length(buf)-5, 2)
    m.static = round(Bool, buf[1])
    m.write_bias_every = round(Int64, buf[2])
    m.bin_width = buf[3]
    m.cvlims = (buf[4], buf[5])
    m.bin_vals = buf[6:5+len_vals]
    m.values = buf[6+len_vals:length(buf)]
    return nothing
end
