"""
Weighting schemes based on the ones compared in \\
https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867
"""
calc_weights(::Nothing, args...; kwargs...) = nothing
calc_weights(::NoBias, args...; kwargs...) = nothing

function calc_weights(b::Bias, cv, itrj; mpi_multi_sim=false)
    calc_weights(b.datafile, b, cv, itrj; mpi_multi_sim=mpi_multi_sim)
    return nothing
end

function calc_weights(b::Vector{<:Bias}, cv, itrj)
    for i in eachindex(b)
        calc_weights(b[i].datafile, b[i], cv[i], itrj)
    end

    return nothing
end

function calc_weights(datafiles, b::Vector{<:Bias}, cv, itrj)
    for i in eachindex(b)
        calc_weights(datafiles[i], b[i], cv[i], itrj)
    end

    return nothing
end

function calc_weights(
    datafile, b::Bias{TCV,TS,TB}, cv, itrj; mpi_multi_sim=false
) where {TCV,TS,TB}
    mpi_amroot(mpi_comm_instance()) || return nothing
    
    for method in b.kinds_of_weights
        w = 0.0
        
        for (icv, bias) in enumerate(b.bias)
            w += calc_weight(bias, cv[icv], method)
        end

        @level1("$itrj\t$(string(cv))\t$(w) # cv weight_$method")
    end

    if datafile != ""
        _filename = if mpi_multi_sim
            set_ext!(datafile)
        else
            datafile
        end

        fp = fopen(_filename, "a")
        printf(fp, "%-11i", itrj)

        for i in eachindex(cv)
            printf(fp, "%+-25.15E", cv[i])
        end

        for method in b.kinds_of_weights
            w = 0.0

            for (icv, bias) in enumerate(b.bias)
                w += calc_weight(bias, cv[icv], method)
            end

            printf(fp, "%-25.15E", w)
        end

        newline(fp)
        fclose(fp)
    end

    return nothing
end

function calc_weight(p::Parametric, cv, args...)
    return exp(p(cv))
end

function calc_weight(o::OPES, cv, args...)
    calculate!(o, cv)
    w = o.current_weight
    return w
end

function calc_weight(o::OPESmultithermal, cv, args...)
    current_bias = o(cv)
    # w = [exp(-o.λ[i]*cv + current_bias) for i in eachindex(o.λ)]
    w = exp(current_bias)
    return w
end

function calc_weight(m::Metadynamics, cv, weight_method)
    w = 0.0

    if weight_method == "tiwari" # average over exp(V) in denom
        w += calc_weight_tiwari(m, cv)
    elseif weight_method == "balanced_exp" # average over V in denom
        w += calc_weight_balanced_exp(m, cv)
    elseif weight_method == "branduardi" # constant bias
        w += exp(m(cv))
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
