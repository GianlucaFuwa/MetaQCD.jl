"""
    Metadynamics(p::ParameterSet; verbose=Verbose1(), instance=1)

Create an instance of a Metadynamics bias using the parameters given in `p`. \\
`verbose` is used to print all parameters out as they are explicitly defined in the
constructor.

# Specifiable parameters
`symmetric::Bool = true` - If `true`, the bias is built symmetrically by updating for both cv and
-cv at every update-iteration \\
`stride::Int64 = 1` - Number of iterations between updates; must be >0 \\
`cvlims::NTuple{2, Float64} = (-6, 6)` - Minimum and maximum of the explorable cv-space;
must be ordered \\
`biasfactor::Float64 = Inf` - Biasfactor for well-tempered Metadynamics; must be >1 \\
`bin_width::Float64 = 0.1` - Width of bins in histogram; must be >0 \\
`weight::Float64 = 0.01` - (Starting) Height of added Gaussians; must be positive \\
`penalty_weight::Float64 = 1000` - Penalty when cv is outside of `cvlims`; must be positive \\
"""
struct Metadynamics <: AbstractBias
    symmetric::Bool
    stride::Int64
    cvlims::NTuple{2, Float64}

    biasfactor::Float64
    bin_width::Float64
    weight::Float64
    penalty_weight::Float64

    bin_vals::Vector{Float64}
    values::Vector{Float64}
end

function Metadynamics(p::ParameterSet; instance=1)
    symmetric = p.symmetric
    stride = p.stride
    @level1("|  STRIDE: $(stride)")
    @assert stride>0 "STRIDE must be >0"

    if instance == 0
        bin_vals, values = metad_from_file(p, "")
    elseif instance > length(p.usebiases)
        bin_vals, values = metad_from_file(p, "")
    else
        bin_vals, values = metad_from_file(p, p.usebiases[instance])
    end

    @level1("|  CVLIMS: $(p.cvlims)")
    @assert issorted(p.cvlims) "CVLIMS must be sorted from low to high"

    @level1("|  BIN_WIDTH: $(p.bin_width)")
    @assert p.bin_width > 0 "BIN_WIDTH must be > 0"

    @level1("|  META_WEIGHT: $(p.meta_weight)")
    @assert p.meta_weight > 0 "META_WEIGHT must be > 0, try \"is_static=true\""

    @level1("|  PENALTY_WEIGHT: $(p.penalty_weight)")

    biasfactor = p.biasfactor
    @level1("|  BIASFACTOR: $(biasfactor)")
    @assert biasfactor > 1 "BIASFACTOR must be > 1"
    return Metadynamics(symmetric, stride,
                        p.cvlims, biasfactor, p.bin_width, p.meta_weight, p.penalty_weight,
                        bin_vals, values)
end

Base.length(m::Metadynamics) = length(m.values)
Base.eachindex(m::Metadynamics) = eachindex(m.values)
Base.lastindex(m::Metadynamics) = lastindex(m.values)

function Base.setindex!(m::Metadynamics, v, i)
    m.values[i] = v
end

@inline function Base.getindex(m::Metadynamics, i)
    return m.values[i]
end

@inline function index(m::Metadynamics, cv)
    idx = (cv - m.cvlims[1]) / m.bin_width + 0.5
    return round(Int64, idx, RoundNearestTiesAway)
end

function update!(m::Metadynamics, cv, args...)
    for cvᵢ in cv
        if in_bounds(cvᵢ, m.cvlims[1], m.cvlims[2])
            for (idx, bin_val) in enumerate(m.bin_vals)
                wt = exp(-m[idx] / m.biasfactor)
                m[idx] += m.weight * wt * exp(-0.5(cvᵢ - bin_val)^2 / m.bin_width^2)
            end
        end

        if m.symmetric
            if in_bounds(-cvᵢ, m.cvlims[1], m.cvlims[2])
                for (idx, bin_val) in enumerate(m.bin_vals)
                    wt = exp(-m[idx] / m.biasfactor)
                    m[idx] += m.weight * wt * exp(-0.5(-cvᵢ - bin_val)^2 / m.bin_width^2)
                end
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
        return m[idx] * (1 - interpolation_constant) + interpolation_constant * m[idx+1]
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
    num = -m(cv+2bw) + 8m(cv+bw) - 8m(cv-bw) + m(cv-2bw)
    denom = 12bw
    return num / denom
end

function clear!(m::Metadynamics)
    @batch for i in eachindex(m)
        m[i] = 0.0
    end

    return nothing
end

write_to_file(::Metadynamics, ::Nothing) = nothing

function write_to_file(m::Metadynamics, filename::String)
    filename=="" && return nothing
    (tmppath, tmpio) = mktemp() # open temporary file at arbitrary location in storage
    println(tmpio, "$(rpad("CV", 7))\t$(rpad("V(CV)", 7))")

    for i in eachindex(m)
        println(tmpio, "$(rpad(m.bin_vals[i], 7, "0"))\t$(rpad(m.values[i], 7, "0"))")
    end

    close(tmpio)
    mv(tmppath, filename, force=true) # replace bias file with temporary file
    return nothing
end

function metad_from_file(p::ParameterSet, usebias)
    if usebias == ""
        bin_vals = range(p.cvlims[1], p.cvlims[2], step=p.bin_width)
        values = zero(bin_vals)
        println("\t>> initialized as zeros")
        return bin_vals, values
    else
        values, _ = readdlm(usebias, Float64, header=true)
        bin_vals = range(p.cvlims[1], p.cvlims[2], step=p.bin_width)
        @assert length(values[:, 2])==length(bin_vals) "your bias doesn't match parameters"
        println("\t>> initialized from \"$(usebias)\"")
        return bin_vals, values[:, 2]
    end
end
