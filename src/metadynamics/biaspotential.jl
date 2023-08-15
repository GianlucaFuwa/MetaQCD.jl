struct BiasPotential{TCV,TG,TS} <: AbstractBiasPotential
    kind_of_cv::TCV
    smearing::TS
    symmetric::Bool

    CVlims::NTuple{2, Float64}
    bin_width::Float64
    weight::Float64
    penalty_weight::Float64
    wt_factor::Float64
    is_static::Bool

    bin_vals::Vector{Float64}
    values::Vector{Float64}
    fp::Union{Nothing, String}
    weight_fp::Union{Nothing, IOStream}
    kinds_of_weights::Union{Nothing, Vector{String}}

    function BiasPotential(p::ParameterSet, U::TG; instance=1, has_fp=true) where {TG}
        TCV = get_cvtype_from_parameters(p)
        smearing = StoutSmearing(U, p.numsmears_for_cv, p.rhostout_for_cv)

        if instance == 0
            bin_vals, values = potential_from_file(p, nothing)
        elseif p.usebiases !== nothing && instance > length(p.usebiases)
            bin_vals, values = potential_from_file(p, nothing)
        elseif p.usebiases === nothing
            bin_vals, values = potential_from_file(p, nothing)
        else
            bin_vals, values = potential_from_file(p, p.usebiases[instance])
        end

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

        return new{TCV,TG,typeof(smearing)}(
            TCV(), smearing, p.symmetric,
            p.cvlims, p.bin_width, p.meta_weight, p.penalty_weight, wt_factor, is_static,
            bin_vals, values, fp, weight_fp, kinds_of_weights,
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

        return new{typeof(kind_of_cv),TG,typeof(smearing)}(
            kind_of_cv, smearing, symmetric,
            CVlims, bin_width, weight, penalty_weight, is_static,
            values, bin_vals, fp, weight_fp, kinds_of_weights,
        )
    end
end

(b::BiasPotential)(cv) = return_potential(b, cv)

Base.length(b::BiasPotential) = length(b.values)
Base.eachindex(b::BiasPotential) = eachindex(b.values)
Base.lastindex(b::BiasPotential) = length(b.values)

function Base.setindex!(b::T, v, i) where {T<:BiasPotential}
    b.values[i] = v
end

@inline function Base.getindex(b::T, i) where {T<:BiasPotential}
    return b.values[i]
end

@inline function kind_of_cv(::BiasPotential{TCV,TG,TS}) where {TCV,TG,TS}
    return TCV()
end

@inline function index(b::T, cv) where {T<:BiasPotential}
    idx = (cv - b.CVlims[1]) / b.bin_width + 0.5
    return round(Int64, idx, RoundNearestTiesAway)
end

function return_potential(b::T, cv) where {T<:BiasPotential}
    if b.CVlims[1] <= cv < b.CVlims[2]-b.bin_width
        idx = index(b, cv)
        interpolation_constant = (cv - b.bin_vals[idx]) / b.bin_width
        return b[idx] * (1 - interpolation_constant) + interpolation_constant * b[idx+1]
    elseif cv < b.CVlims[1]
        penalty = b[1] + b.penalty_weight * min((cv - b.CVlims[1])^2, (cv - b.CVlims[2])^2)
        return penalty
    else
        penalty = b[end] + b.penalty_weight * min((cv - b.CVlims[1])^2, (cv - b.CVlims[2]+b.bin_width)^2)
        return penalty
    end
end

function clear!(b::T) where {T<:BiasPotential}
    @batch for i in eachindex(b)
        b[i] = 0.0
    end

    return nothing
end

function write_to_file(b::T; force=false) where {T<:BiasPotential}
    # If the potential is static we dont have to write it apart from the the time it
    # is initialized, so we introduce a "force" keyword to overwrite the static-ness
    (b.is_static && !force) && return nothing
    (b.fp === nothing) && return nothing
    (tmppath, tmpio) = mktemp() # open temporary file at arbitrary location in storage
    println(tmpio, "$(rpad("CV", 7))\t$(rpad("V(CV)", 7))")

    for i in eachindex(b)
        println(tmpio, "$(rpad(b.bin_vals[i], 7, "0"))\t$(rpad(b.values[i], 7, "0"))")
    end

    close(tmpio)
    mv(tmppath, b.fp, force = true) # replace bias file with temporary file
    return nothing
end
