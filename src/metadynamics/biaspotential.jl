struct BiasPotential{TG} <: AbstractBiasPotential
    kind_of_cv::String
    smearing::StoutSmearing{TG}
    symmetric::Bool

    CVlims::NTuple{2, Float64}
    bin_width::Float64
    weight::Float64
    penalty_weight::Float64
    wt_factor::Float64
    is_static::Bool

    values::Vector{Float64}
    bin_vals::Vector{Float64}
    fp::Union{Nothing, String}
    weight_fp::Union{Nothing, IOStream}
    kinds_of_weights::Union{Nothing, Vector{String}}

    function BiasPotential(
        p::ParameterSet,
        U::TG;
        instance = 1,
        has_fp::Bool = true,
    ) where {TG}
        smearing = StoutSmearing(U, p.numsmears_for_cv, p.ρstout_for_cv)

        if instance == 0
            values = potential_from_file(p, nothing)
        elseif p.usebiases !== nothing && instance > length(p.usebiases)
            values = potential_from_file(p, nothing)
        elseif p.usebiases === nothing
            values = potential_from_file(p, nothing)
        else
            values = potential_from_file(p, p.usebiases[instance])
        end

        bin_vals = range(p.CVlims[1], p.CVlims[2], step = p.bin_width)

        if has_fp == true && instance > 0
            fp = p.biasdir * "/stream_$instance.txt"
            weight_fp = open(p.measuredir * "/meta_weight_$instance.txt", "w")
            kinds_of_weights = p.kinds_of_weights
            header = rpad("itrj", 9, " ")

            for name in kinds_of_weights
                name_str = "weight_$name"
                header *= "\t$(rpad(name_str, 22, " "))"
            end

            println(weight_fp, header)
        else
            fp = nothing
            weight_fp = nothing
            kinds_of_weights = nothing
        end

        wt_factor = p.wt_factor
        is_static = instance == 0 ? true : p.is_static[instance]

        return new{TG}(
            p.kind_of_cv, smearing, p.symmetric,
            p.CVlims, p.bin_width, p.meta_weight, p.penalty_weight, wt_factor, is_static,
            values, bin_vals, fp, weight_fp, kinds_of_weights,
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
        values = zeros(round(Int64, (CVlims[2]-CVlims[1]) / bin_width) + 1)
        bin_vals = range(CVlims[1], CVlims[2], step = bin_width)
        # exceeded_count = 0
        fp = nothing
        weight_fp = nothing
        kinds_of_weights = nothing

        return new{TG}(
            kind_of_cv, smearing, symmetric,
            CVlims, bin_width, weight, penalty_weight, is_static,
            values, bin_vals, fp, weight_fp, kinds_of_weights,
        )
    end
end

function Base.length(b::T) where {T <: BiasPotential}
    return length(b.values)
end

function Base.eachindex(b::T) where {T <: BiasPotential}
    return eachindex(b.values)
end

function Base.setindex!(b::T, v, i) where {T <: BiasPotential}
    b.values[i] = v
end

@inline function Base.getindex(b::T, i) where {T <: BiasPotential}
    return b.values[i]
end

function Base.lastindex(b::T) where {T <: BiasPotential}
    return length(b.values)
end

@inline function index(b::T, cv) where {T <: BiasPotential}
    grid_index = (cv - b.CVlims[1]) / b.bin_width + 0.5
    return round(Int64, grid_index, RoundNearestTiesAway)
end

function return_potential(b::T, cv) where {T <: BiasPotential}
    if b.CVlims[1] <= cv < b.CVlims[2]
        grid_index = index(b, cv)
        return b[grid_index]
    elseif cv < b.CVlims[1]
        penalty = b[1] + b.penalty_weight * min((cv - b.CVlims[1])^2, (cv - b.CVlims[2])^2)
        return penalty
    else
        penalty= b[end] + b.penalty_weight * min((cv - b.CVlims[1])^2, (cv - b.CVlims[2])^2)
        return penalty
    end
end

function (b::BiasPotential)(cv)
    return return_potential(b, cv)
end

function clear!(b::T) where {T <: BiasPotential}
    @batch for i in eachindex(b)
        b[i] = 0.0
    end

    return nothing
end
