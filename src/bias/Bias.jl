"""
    BiasModule

Module containing all things relevant to Metadynamics like the bias potential,
overloaded methods, the potential derivative (scalar) and the calculation of weights

    BiasPotential{TG} -> Holds the bias potential and its parameters, its I/O and most
                         importantly the smearing struct necessary for the calculation of
                         the charge-force
                         TG specifies the kind of Gaugefield, i.e. gauge action, it is
                         associated with
"""
module BiasModule
    using DelimitedFiles
    using Polyester
    using Printf
    using Statistics
	using ..Parameters: ParameterSet

    import ..Gaugefields: AbstractGaugeAction, Gaugefield, Plaquette, Clover
	import ..Measurements: top_charge
	import ..Smearing: NoSmearing, StoutSmearing, calc_smearedU!

	struct BiasEnabled end
	struct BiasDisabled end

    abstract type AbstractBias end

    struct Bias{TCV,TS,TB}
        kind_of_cv::TCV
        smearing::TS
        bias::TB
        biasfile::Union{Nothing, String}
        write_bias_every::Int64
        kinds_of_weights::Union{Nothing, Vector{String}}
        weight_fp::Union{Nothing, IOStream}

        function Bias(p::ParameterSet, U; verbose=nothing, instance=1, has_fp=true)
            TCV = get_cvtype_from_parameters(p)
            smearing = StoutSmearing(U, p.numsmears_for_cv, p.rhostout_for_cv)

            if p.kind_of_bias == "metad"
                bias = Metadynamics(p; instance=instance)
            elseif p.kind_of_bias == "opes"
                bias = OPES(p; instance=instance, verbose=verbose)
            else
                error("kind_of_bias $(p.kind_of_bias) not supported. Try metad or opes")
            end

            if has_fp == true && instance > 0
                biasfile = p.biasdir * "/stream_$instance.txt"
                weight_fp = open(p.measuredir * "/meta_weight_$instance.txt", "w")
                kinds_of_weights = p.kinds_of_weights
                header = rpad("itrj", 9, " ")

                for name in kinds_of_weights
                    name_str = "weight_$name"
                    header *= "\t$(rpad(name_str, 22, " "))"
                end

                println(weight_fp, header)
            else
                biasfile = nothing
                kinds_of_weights = nothing
                weight_fp = nothing
            end

            write_bias_every = p.write_bias_every===nothing ? 0 : p.write_bias_every
            print_verbose1(verbose, "\t>> WRITE_BIAS_EVERY = $(write_bias_every)")
            @assert (write_state_every==0) || (write_state_every>=p.stride)

            return new{TCV,typeof(smearing),typeof(bias)}(
                TCV(),
                smearing,
                bias, biasfile, write_bias_every,
                kinds_of_weights, weight_fp,
            )
        end
    end

    include("metadynamics.jl")
    include("opes.jl")

    update_bias!(::Nothing, args...) = nothing

    function update_bias!(b::Bias, values, itrj, write)
        for cv in values
            update!(b.bias, cv, itrj)
        end

        if b.biasfile !== nothing
            # If the potential is static we dont have to write it apart from the the time it
            # is initialized, so we introduce a "force" keyword to overwrite the static-ness
            (b.is_static && !write) && return nothing
            write_to_file(b.bias, b.biasfile)
        end
        return nothing
    end

    recalc_CV!(::Gaugefield, ::Nothing) = nothing

    function recalc_CV!(U, b::Bias)
        CV_new = calc_CV(U, b)
        U.CV = CV_new
        return nothing
    end

    function calc_CV(U, ::Bias{TCV,TS,TB}) where {TCV,TS<:NoSmearing,TB}
        return top_charge(TCV(), U)
    end

    function calc_CV(U, b::Bias{TCV,TS,TB}) where {TCV,TS,TB}
        calc_smearedU!(b.smearing, U)
        fully_smeared_U = b.smearing.Usmeared_multi[end]
        CV_new = top_charge(TCV(), fully_smeared_U)
        return CV_new
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

    kind_of_cv(b::Bias) = b.kind_of_cv

    function in_bounds(cv, bounds)
        bounds[1] <= cv <= bounds[2] && return true
        return false
    end

    include("weights.jl")

end
