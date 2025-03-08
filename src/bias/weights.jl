"""
Weighting schemes based on the ones compared in \\
https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867
"""
calc_weights(::Nothing, args...; kwargs...) = nothing
calc_weights(::NoBias, args...; kwargs...) = nothing

function calc_weights(b::Bias, cv, itrj, myinstance=mpi_myrank(); mpi_multi_sim=false)
    calc_weights(b.datafile, b, cv, itrj, myinstance; mpi_multi_sim=mpi_multi_sim)
    return nothing
end

function calc_weights(b::Vector{<:Bias}, cv, itrj)
    for i in eachindex(b)
        calc_weights(b[i].datafile, b[i], cv[i], itrj)
    end

    return nothing
end

function calc_weights(filenames, b::Vector{<:Bias}, cv, itrj)
    for i in eachindex(b)
        calc_weights(filenames[i], b[i], cv[i], itrj)
    end

    return nothing
end

function calc_weights(
    filename, b::Bias{TCV,TS,TB}, cv, itrj, myinstance=mpi_myrank();
    mpi_multi_sim=false
) where {TCV,TS,TB}
    if isnothing(filename) || !isfile(filename)
        for method in b.kinds_of_weights
            w = calc_weight(b.bias, cv, method)
            @level1("$itrj\t$cv\t$w # cv weight_$method")
        end
    else
        datafile = if mpi_multi_sim
            set_ext!(filename, myinstance)
        else
            filename
        end
        @level1 "Data file $(myinstance): $(datafile)"

        fp = fopen(datafile, "a")
        printf(fp, "%-11i", itrj)
        printf(fp, "%+-25.15E", cv)

        for method in b.kinds_of_weights
            w = calc_weight(b.bias, cv, method)
            printf(fp, "%-25.15E", w)
        end

        newline(fp)
        fclose(fp)
    end

    return nothing
end

calc_weight(p::Parametric, cv, args...) = exp(p(cv))

function calc_weight(o::OPES, cv, args...)
    calculate!(o, cv)
    return o.current_weight
end

function calc_weight(m::Metadynamics, cv, weight_method)
    w = 0.0

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
    norm = mean(exp(val) for val in m.values)
    w = exp(m(cv)) / norm
    return w
end

function calc_weight_balanced_exp(m::Metadynamics, cv)
    norm = exp(mean(val for val in m.values))
    w = exp(m(cv)) / norm
    return w
end
