"""
Weighting schemes based on the ones compared in \\
https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867
"""
calc_weights(::Nothing, args...) = nothing
calc_weights(::NoBias, args...) = nothing

function calc_weights(b::Vector{<:Bias}, cv, itrj)
    for i in eachindex(b)
        calc_weights(b[i], cv[i], itrj)
    end

    return nothing
end

function calc_weights(b::Bias{TCV,TS,TB,T}, cv, itrj) where {TCV,TS,TB,T}
    b.kinds_of_weights === nothing && return nothing
    str = @sprintf("%-9i\t%+-22.15E", itrj, cv)
    for method in b.kinds_of_weights
        w = calc_weight(b.bias, cv, method)
        str *= @sprintf("\t%-22.15E", w)
    end

    @level1(str * " # cv weight")

    if T ≢ Nothing
        println(b.fp, str)
        flush(b.fp)
    end

    return nothing
end

calc_weight(p::Parametric, cv, args...) = exp(p(cv))

function calc_weight(o::OPES, cv, args...)
    calculate!(o, cv)
    return o.current_weight
end

function calc_weight(m::Metadynamics, cv, weight_method)
    if weight_method == "tiwari" # average over exp(V) in denom
        w = calc_weight_tiwari(m, cv)
    elseif weight_method == "balanced_exp" # average over V in denom
        w = calc_weight_balanced_exp(m, cv)
    elseif weight_method == "branduardi" # constant bias
        w = exp(m(cv))
    else
        error("MetaD weighting method \"$weight_method\" not supported")
    end

    return w
end

function calc_weight_tiwari(m::Metadynamics, cv)
    norm = mean(exp.(m.values))
    w = exp(m(cv)) / norm
    return w
end

function calc_weight_balanced_exp(m::Metadynamics, cv)
    norm = exp(mean(m.values))
    w = exp(m(cv)) / norm
    return w
end
