module BiasModule
    using DelimitedFiles
    using Polyester
    using Printf
    using Statistics
	using ..Parameters: ParameterSet
    using ..Output

    import ..Gaugefields: AbstractGaugeAction, Gaugefield, Plaquette, Clover
	import ..Measurements: top_charge
	import ..Smearing: NoSmearing, StoutSmearing, calc_smearedU!

	struct BiasEnabled end # probably unnecessary
	struct BiasDisabled end

    abstract type AbstractBias end

    """
        Bias(p::ParameterSet, U::Gaugefield; verbose=nothing, instance=1, has_fp=true)

    Container that holds general parameters of bias enhanced sampling, like the kind of CV,
    its smearing and filenames/-pointers relevant to the bias. Also holds the specific kind
    of bias (`Metadynamics` or `OPES` for now). \\
    The `instance` keyword is used in case of PT-MetaD and multiple walkers assign the
    correct `usebias` to each stream. `has_fp` indicates whether the stream prints to file
    at any point, since only rank 0 should print in case of MPI usage.
    """
    struct Bias{TCV,TS,TB}
        kind_of_cv::TCV
        smearing::TS
        is_static::Bool
        bias::TB
        biasfile::Union{Nothing, String}
        write_bias_every::Int64
        kinds_of_weights::Union{Nothing, Vector{String}}
        weight_fp::Union{Nothing, IOStream}

        function Bias(p::ParameterSet, U; verbose=Verbose1(), instance=1, has_fp=true)
            TCV = get_cvtype_from_parameters(p)
            smearing = StoutSmearing(U, p.numsmears_for_cv, p.rhostout_for_cv)
            is_static = instance==0 ? true : p.is_static[instance]
            println_verbose1(verbose, ">> Bias is $(ifelse(is_static, "static", "dynamic"))")

            if p.kind_of_bias == "metad"
                bias = Metadynamics(p; instance=instance, verbose=verbose)
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

            println_verbose1(verbose, "\t>> BIASFILE = $(biasfile)")
            write_bias_every = p.write_bias_every===nothing ? 0 : p.write_bias_every
            println_verbose1(verbose, "\t>> WRITE_BIAS_EVERY = $(write_bias_every)")
            @assert (write_bias_every==0) || (write_bias_every>=p.stride)
            println_verbose1(verbose)

            # write to file after construction to make sure nothing went wrong
            biasfile!==nothing && write_to_file(bias, biasfile)

            return new{TCV,typeof(smearing),typeof(bias)}(
                TCV(), smearing, is_static,
                bias, biasfile, write_bias_every,
                kinds_of_weights, weight_fp,
            )
        end
    end

    (b::Bias)(cv) = b.bias(cv)

    include("metadynamics.jl")
    include("opes.jl")

    update_bias!(::Nothing, args...) = nothing

    function update_bias!(b::Bias, values, itrj, write)
        b.is_static && return nothing

        for cv in values
            update!(b.bias, cv, itrj)
        end

        (b.biasfile!==nothing && write) && write_to_file(b.bias, b.biasfile)
        return nothing
    end

    recalc_CV!(::Gaugefield, ::Nothing) = nothing

    function recalc_CV!(U::Gaugefield, b::Bias)
        CV_new = calc_CV(U, b)
        U.CV = CV_new
        return nothing
    end

    function recalc_CV!(U::Vector{TG}, b::Vector{TB}) where {TG<:Gaugefield, TB<:Bias}
        for i in eachindex(U)
            recalc_CV!(U[i], b[i])
        end
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

    ∂V∂Q(b::Bias, cv) = ∂V∂Q(b.bias, cv)

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

    function in_bounds(cv, lb, ub)
        lb <= cv <= ub && return true
        return false
    end

    include("weights.jl")

end
