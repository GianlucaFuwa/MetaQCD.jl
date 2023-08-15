"""
Weighting schemes based on the ones compared in \\
https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867
"""
calc_weights(::Nothing, cv, itrj) = nothing

function calc_weights(b, cv, itrj)
    b === nothing && return nothing

    printstring = "$(rpad(itrj, 9, " "))\t"

    if b.weight_fp !== nothing
        for method in b.kinds_of_weights
            w = calc_weight(b, cv, method)
            w_str = @sprintf("%.15E", w)
            printstring *= "$w_str\t"
        end

        println(b.weight_fp, printstring)
        flush(b.weight_fp)
    end

    return nothing
end

function calc_weight(b::T, cv, weight_method) where {T<:BiasPotential}
    if weight_method == "tiwari" # average over exp(V) in denom
        w = calc_weight_tiwari(b, cv)
    elseif weight_method == "balanced_exp" # average over V in denom
        w = calc_weight_balanced_exp(b, cv)
    elseif weight_method == "branduardi" # constant bias
        w = exp(b(cv))
    else
        error("MetaD weighting method \"$weight_method\" not supported")
    end

    return w
end

function calc_weight_tiwari(b, cv)
    w = exp(b(cv)) / mean(exp.(b.values))
    return w
end

function calc_weight_balanced_exp(b, cv)
    w = exp(b(cv)) / exp(mean(b.values))
    return w
end
