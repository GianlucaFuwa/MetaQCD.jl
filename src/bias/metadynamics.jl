struct Metadynamics <: AbstractBias
    is_static::Bool
    symmetric::Bool

    CVlims::NTuple{2, Float64}
    biasfactor::Float64
    bin_width::Float64
    weight::Float64
    penalty_weight::Float64

    bin_vals::Vector{Float64}
    values::Vector{Float64}

    function Metadynamics(p::ParameterSet; instance=1)
        is_static = instance == 0 ? true : p.is_static[instance]
        symmetric = p.symmetric

        if instance == 0
            bin_vals, values = metad_from_file(p, nothing)
        elseif p.usebiases !== nothing && instance > length(p.usebiases)
            bin_vals, values = metad_from_file(p, nothing)
        elseif p.usebiases === nothing
            bin_vals, values = metad_from_file(p, nothing)
        else
            bin_vals, values = metad_from_file(p, p.usebiases[instance])
        end

        biasfactor = p.biasfactor

        return new(
            is_static, symmetric,
            p.cvlims, p.bin_width, p.meta_weight, p.penalty_weight, biasfactor,
            bin_vals, values,
        )
    end
end

Base.length(m::Metadynamics) = length(m.values)
Base.eachindex(m::Metadynamics) = eachindex(m.values)
Base.lastindex(m::Metadynamics) = length(m.values)

function Base.setindex!(m::Metadynamics, v, i)
    m.values[i] = v
end

@inline function Base.getindex(m::Metadynamics, i)
    return m.values[i]
end

@inline function index(m::Metadynamics, cv)
    idx = (cv - m.CVlims[1]) / m.bin_width + 0.5
    return round(Int64, idx, RoundNearestTiesAway)
end

function update!(m::Metadynamics, cv, itrj)
    (m.is_static==true || itrj%m.stride!=0) && return nothing

    if in_bounds(cv, m.CVlims)
        for (idx, current_bin) in enumerate(m.bin_vals)
            wt = exp(-m[idx] / m.biasfactor)
            m[idx] += m.weight * wt * exp(-0.5(cv - current_bin)^2 / m.bin_width^2)
        end
    end

    if m.symmetric
        if in_bounds(cv, m.CVlims)
            for (idx, current_bin) in enumerate(m.bin_vals)
                wt = exp(-m[idx] / m.biasfactor)
                m[idx] += m.weight * wt * exp(-0.5(-cv - current_bin)^2 / m.bin_width^2)
            end
        end
    end

    return nothing
end

(m::Metadynamics)(cv) = return_potential(m, cv)

function return_potential(m::Metadynamics, cv)
    bw = m.bin_width
    pen = m.penalty_weight

    if in_bounds(cv, m.CVlims)
        idx = index(m, cv)
        interpolation_constant = (cv - m.bin_vals[idx]) / bw
        return m[idx] * (1 - interpolation_constant) + interpolation_constant * m[idx+1]
    elseif cv < m.CVlims[1]
        penalty = m[1] + pen * min((cv - m.CVlims[1])^2, (cv - m.CVlims[2])^2)
        return penalty
    else
        penalty = m[end] + pen * min((cv - m.CVlims[1])^2, (cv - m.CVlims[2]+bw)^2)
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

function write_to_file(m::Metadynamics, filename)
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
    if usebias === nothing
        bin_vals = range(p.cvlims[1], p.cvlims[2], step = p.bin_width)
        values = zero(bin_vals)
        return bin_vals, values
    else
        values, _ = readdlm(usebias, Float64, header=true)
        bin_vals = range(p.cvlims[1], p.cvlims[2], step = p.bin_width)
        @assert length(values[:, 2])==length(bin_vals) "your bias doesn't match parameters"
        return bin_vals, values[:, 2]
    end
end
