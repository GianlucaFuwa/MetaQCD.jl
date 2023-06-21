struct BiasPotential{TG} <: AbstractBiasPotential
    kind_of_cv::String
    smearing::StoutSmearing{TG}
    symmetric::Bool

    CVlims::NTuple{2, Float64}
    bin_width::Float64
    weight::Float64
    penalty_weight::Float64
    is_static::Bool

    values::Vector{Float64}
    bin_vals::Vector{Float64}
    fp::Union{Nothing, IOStream}

    function BiasPotential(
        p::Params,
        U::Gaugefield{TG};
        instance = 1,
        biasfile = nothing,
        has_fp::Bool = true,
    ) where {TG}
        smearing = StoutSmearing(U, p.numsmears_for_cv, p.ρstout_for_cv)

        if instance == 0
            values = potential_from_file(p, nothing)
        else
            if p.usebiases === nothing
                values = potential_from_file(p, nothing)
            else
                values = potential_from_file(p, p.usebiases[instance])
            end
        end

        bin_vals = range(p.CVlims[1], p.CVlims[2], step = p.bin_width)

        if has_fp == true
            if biasfile === nothing
                fp = open(p.biasdir * "/Stream_$instance.txt", "w")
            else
                fp = open(p.biasdir * "/" * biasfile * ".txt", "w")
            end
        else
            fp = nothing
        end

        is_static = instance == 0 ? true : p.is_static[instance]

        return new{TG}(
            p.kind_of_cv, smearing, p.symmetric,
            p.CVlims, p.bin_width, p.meta_weight, p.penalty_weight, is_static,
            values, bin_vals, fp,
        )
    end

    function BiasPotential(
        U::Gaugefield{TG},
        kind_of_cv,
        numsmear,
        ρstout,
        symmetric,
        CVlims,
        bin_width,
        weight,
        penalty_weight,
    ) where {TG}
        smearing = StoutSmearing(U, numsmear, ρstout)
        is_static = false
        values = zeros(round(Int64, (CVlims[2]-CVlims[1]) / bin_width, RoundNearestTiesAway) + 1)
        bin_vals = range(CVlims[1], CVlims[2], step = bin_width)
        # exceeded_count = 0
        fp = nothing

        return new{TG}(
            kind_of_cv, smearing, symmetric,
            CVlims, bin_width, weight, penalty_weight, is_static,
            values, bin_vals, fp,
        )
    end
end

function Base.length(b::T) where {T <: BiasPotential}
    return length(b.values)
end

function Base.eachindex(b::T) where {T <: BiasPotential}
    return eachindex(b.values)
end

function Base.flush(b::T) where {T <: BiasPotential}
    if b.fp !== nothing
        flush(b.fp)
    end
end

function Base.seekstart(b::T) where {T <: BiasPotential}
    if b.fp !== nothing
        seekstart(b.fp)
    end
end

function Base.setindex!(b::T, v, i) where {T <: BiasPotential}
    b.values[i] = v
end

@inline function Base.getindex(b::T, i) where {T <: BiasPotential}
    return b.values[i]
end

@inline function index(b::T, cv) where {T <: BiasPotential}
    grid_index = (cv - b.CVlims[1]) / b.bin_width + 0.5
    return round(Int64, grid_index, RoundNearestTiesAway)
end

function update_bias!(b::T, cv) where {T <: BiasPotential}
    grid_index = index(b, cv)

    if 1 <= grid_index <= length(b.values)
        for (idx, current_bin) in enumerate(b.bin_vals)
            b[idx] += b.weight * exp(-0.5(cv - current_bin)^2 / b.bin_width^2)
        end
    else
        # b.exceeded_count += 1
    end

    return nothing
end

function (b::BiasPotential)(cv)
    return return_potential(b, cv)
end

function return_potential(b::T, cv) where {T <: BiasPotential}
    if b.CVlims[1] <= cv < b.CVlims[2]
        grid_index = index(b, cv)
        return b[grid_index]
    else
        penalty = b.penalty_weight * (
            0.1 + min((cv - b.CVlims[1])^2, (cv - b.CVlims[2])^2)
        )
        return penalty
    end
end

function clear!(b::T) where {T <: BiasPotential}
    @batch for i in eachindex(b)
        b[i] = 0.0
    end

    return nothing
end

"""
Approximate ∂V/∂Q by use of the five-point stencil
"""
function ∂V∂Q(b::T, cv) where {T <: BiasPotential}
    bin_width = b.bin_width
    num =
        -b(cv + 2 * bin_width) +
        8 * b(cv + bin_width) -
        8 * b(cv - bin_width) +
        b(cv - 2 * bin_width)
    denom = 12 * bin_width
    return num / denom
end