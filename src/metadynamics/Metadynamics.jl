"""
    Metadynamics

Module containing all things relevant to Metadynamics like the bias potential,
overloaded methods, the potential derivative (scalar) and the calculation of weights

    BiasPotential{TG} -> Holds the bias potential and its parameters, its I/O and most
                         importantly the smearing struct necessary for the calculation of
                         the charge-force
                         TG specifies the kind of Gaugefield, i.e. gauge action, it is
                         associated with
"""
module Metadynamics
    using DelimitedFiles
    using Polyester
    using Printf
    using Statistics
	using ..Parameters: ParameterSet

    import ..Gaugefields: AbstractGaugeAction, Gaugefield
	import ..Measurements: top_charge
	import ..Smearing: StoutSmearing

	struct MetaEnabled end
	struct MetaDisabled end

    abstract type AbstractBiasPotential end

    include("biaspotential.jl")

	function potential_from_file(p::ParameterSet, usebias)
		if usebias === nothing
			return zero(range(p.CVlims[1], p.CVlims[2], step = p.bin_width))
		else
			values = readdlm(usebias, Float64, skipstart = 1)
			@assert length(values[:, 2]) == length(
                range(p.CVlims[1], p.CVlims[2], step = p.bin_width)
            ) "Length of passed Metapotential doesn't match parameters"
			return values[:, 2]
		end
	end

    function update_bias!(b::T, cv) where {T <: BiasPotential}
        (b.is_static == true) && return nothing
        grid_index = index(b, cv)

        if 1 <= grid_index <= length(b.values)
            for (idx, current_bin) in enumerate(b.bin_vals)
                wt = exp(-b[idx] / b.wt_factor)
                b[idx] += b.weight * wt * exp(-0.5(cv - current_bin)^2 / b.bin_width^2)
            end
        else
            # b.exceeded_count += 1
        end

        return nothing
    end

    update_bias!(::Nothing, cv) = nothing

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

    """
    Weighting schemes based on the ones compared in \\
    https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867
    """
    calc_weights(::Nothing, cv, itrj) = nothing

    function calc_weights(b, cv, itrj)
        b === nothing && return nothing

        printstring = "$itrj\t"

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

    function calc_weight(b::T, cv, weight_method) where {T <: BiasPotential}
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
end
