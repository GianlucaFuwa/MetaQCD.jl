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
	import ..Smearing: StoutSmearing, calc_smearedU!

	struct MetaEnabled end
	struct MetaDisabled end

    abstract type AbstractBiasPotential end

    include("biaspotential.jl")

    function potential_from_file(p::ParameterSet, usebias)
        if usebias === nothing
            return zero(range(p.CVlims[1], p.CVlims[2], step = p.bin_width))
        else
            values, _ = readdlm(usebias, Float64, header = true)
            @assert length(values[:, 2]) == length(
                range(p.CVlims[1], p.CVlims[2], step = p.bin_width)
            ) "Length of passed Metapotential doesn't match parameters"
            return values[:, 2]
        end
    end

    """
    Update bias potential .txt file by creating temporary io where the new valus are \\
    written, which then substitutes the old file
    """
    function write_to_file(b::T; force = false) where {T <: BiasPotential}
        # If the potential is static we dont have to write it apart from the the time it
        # is initialized, so we introduce a "force" keyword to overwrite the static-ness
        (b.is_static && !force) && return nothing
        (b.fp === nothing) && return nothing
        (tmppath, tmpio) = mktemp()
        println(tmpio, "$(rpad("CV", 7))\t$(rpad("V(CV)", 7))")

        for i in eachindex(b)
            println(tmpio, "$(rpad(b.bin_vals[i], 7, "0"))\t$(rpad(b.values[i], 7, "0"))")
        end

        close(tmpio)
        mv(tmppath, b.fp, force = true)
        return nothing
    end

    write_to_file(::Nothing; force = false) = nothing

    function update_bias!(b::T, values; write = true) where {T <: BiasPotential}
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

        write && write_to_file(b)
        return nothing
    end

    update_bias!(::Nothing, cv; write = true) = nothing

    """
    Approximate ∂V/∂Q by use of the five-point stencil
    """
    function ∂V∂Q(b::T, cv) where {T <: BiasPotential}
        bw = b.bin_width
        num = -b(cv + 2bw) + 8b(cv + bw) - 8b(cv - bw) + b(cv - 2bw)
        denom = 12bw
        return num / denom
    end

    function recalc_CV!(U, b::T) where {T <: BiasPotential}
        calc_smearedU!(b.smearing, U)
        fully_smeared_U = b.smearing.Usmeared_multi[end]
        CV_new = top_charge(fully_smeared_U, b.kind_of_cv)
        U.CV = CV_new
        return nothing
    end

    include("weights.jl")

end
