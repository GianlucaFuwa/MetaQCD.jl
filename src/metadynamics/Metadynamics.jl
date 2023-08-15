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
    using MPI
    using Polyester
    using Printf
    using Statistics
	using ..Parameters: ParameterSet

    import ..Gaugefields: AbstractGaugeAction, Gaugefield, Plaquette, Clover
	import ..Measurements: top_charge
	import ..Smearing: NoSmearing, StoutSmearing, calc_smearedU!

	struct MetaEnabled end
	struct MetaDisabled end

    abstract type AbstractBiasPotential end

    include("biaspotential.jl")

    update_bias!(::Nothing, args...) = nothing

    function update_bias!(b::BiasPotential, values, write)
        update_bias!(b, values)
        b.symmetric && update_bias!(b, -values)
        write && write_to_file(b)
        return nothing
    end

    function update_bias!(b, values)
        (b.is_static == true) && return nothing

        for cv in values
            grid_index = index(b, cv)

            if 1 <= grid_index <= length(b.values)
                for (idx, current_bin) in enumerate(b.bin_vals)
                    wt = exp(-b[idx] / b.wt_factor)
                    b[idx] += b.weight * wt * exp(-0.5(cv - current_bin)^2 / b.bin_width^2)
                end
            else
                # b.exceeded_count += 1
            end
        end

        return nothing
    end

    """
    Approximate ∂V/∂Q by use of the five-point stencil
    """
    function ∂V∂Q(b::T, cv) where {T<:BiasPotential}
        bw = b.bin_width
        num = -b(cv+2bw) + 8b(cv+bw) - 8b(cv-bw) + b(cv-2bw)
        denom = 12bw
        return num / denom
    end

    recalc_CV!(::Gaugefield, ::Nothing) = nothing

    function recalc_CV!(U, b::BiasPotential)
        CV_new = calc_CV(U, b)
        U.CV = CV_new
        return nothing
    end

    function calc_CV(U, ::BiasPotential{TCV,TG,TS}) where {TCV,TG,TS<:NoSmearing}
        return top_charge(TCV(), U)
    end

    function calc_CV(U, b::BiasPotential{TCV,TG,TS}) where {TCV,TG,TS}
        calc_smearedU!(b.smearing, U)
        fully_smeared_U = b.smearing.Usmeared_multi[end]
        CV_new = top_charge(TCV(), fully_smeared_U)
        return CV_new
    end

    write_to_file(::Nothing; kwargs...) = nothing

    function potential_from_file(p::ParameterSet, usebias)
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

    function get_cvtype_from_parameters(p::ParameterSet)
        if p.kind_of_cv == "plaquette"
            return Plaquette
        elseif p.kind_of_cv == "clover"
            return Clover
        else
            error("kind of cv \"$(p.kind_of_cv)\" not supported")
        end
    end

    include("weights.jl")

end
