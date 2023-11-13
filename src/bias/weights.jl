"""
Weighting schemes based on the ones compared in \\
https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867
"""
calc_weights(::Nothing, args...) = nothing

function calc_weights(b::Vector{<:Bias}, cv, itrj)
    for i in eachindex(b)
        calc_weights(b[i], cv[i], itrj)
    end

    return nothing
end

function calc_weights(b::Bias, cv, itrj)
    str = @sprintf("%-9i\t%+-22.15E", itrj, cv)
    for method in b.kinds_of_weights
        w = calc_weight(b, cv, method)
        str *= @sprintf("\t%-22.15E", w)
    end
    println(str * " # cv weight\n")

    if b.fp ≢ nothing
        println(b.fp, str * "\n")
        flush(b.fp)
    end
    return nothing
end

calc_weight(b::Parametric, cv, args...) = exp(b(cv))

function calc_weight(b, cv, weight_method)
    if weight_method == "tiwari" # average over exp(V) in denom
        w = calc_weight_tiwari(b.bias, cv)
    elseif weight_method == "balanced_exp" # average over V in denom
        w = calc_weight_balanced_exp(b.bias, cv)
    elseif weight_method == "branduardi" # constant bias
        w = exp(b(cv))
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

function calc_weight_tiwari(o::OPES, cv)
    norm = 0.0
    for kernel in eachkernel(o)
        norm += exp(o(kernel.center))
    end
    norm /= o.nker

    w = exp(o(cv)) / norm
    return w
end

function calc_weight_balanced_exp(m::Metadynamics, cv)
    norm = exp(mean(m.values))
    w = exp(m(cv)) / norm
    return w
end

function calc_weight_balanced_exp(o::OPES, cv)
    norm = 0.0
    for kernel in eachkernel(o)
        norm += o(kernel.center)
    end
    norm /= o.nker

    w = exp(o(cv)) / exp(norm)
    return w
end
