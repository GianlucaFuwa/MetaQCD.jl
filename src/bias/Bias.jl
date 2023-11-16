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

abstract type AbstractBias end

"""
    Bias(p::ParameterSet, U::Gaugefield; verbose=nothing, instance=1, has_fp=true)

Container that holds general parameters of bias enhanced sampling, like the kind of CV,
its smearing and filenames/-pointers relevant to the bias. Also holds the specific kind
of bias (`Metadynamics`, `OPES` or `Parametric` for now). \\
The `instance` keyword is used in case of PT-MetaD and multiple walkers to assign the
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
    fp::Union{Nothing, IOStream}
end

function Bias(p::ParameterSet, U; verbose=Verbose1(), instance=1, has_fp=true)
    TCV = get_cvtype_from_parameters(p)
    smearing = StoutSmearing(U, p.numsmears_for_cv, p.rhostout_for_cv)
    is_static = instance==0 ? true : p.is_static[instance]
    sstr = (is_static || p.kind_of_bias=="parametric") ? "STATIC" : "DYNAMIC"
    println_verbose1(verbose, ">> Bias $instance is $sstr")

    if p.kind_of_bias ∈ ["metad", "metadynamics"]
        bias = Metadynamics(p; instance=instance, verbose=verbose)
    elseif p.kind_of_bias == "opes"
        bias = OPES(p; instance=instance, verbose=verbose)
    elseif p.kind_of_bias == "parametric"
        bias = Parametric(p; instance=instance, verbose=verbose)
    else
        error("kind_of_bias $(p.kind_of_bias) not supported. Try metad or opes")
    end

    println_verbose1(
        verbose,
        "\t>> CV DATA: $TCV WITH $(p.numsmears_for_cv)x$(p.rhostout_for_cv) SMEARS"
    )

    if (has_fp==true) && (p.kind_of_bias!="parametric")
        biasfile = p.biasdir * "/stream_$instance.txt"
        fp = open(p.measuredir * "/bias_data_$instance.txt", "w")
        kinds_of_weights = p.kinds_of_weights
        @printf(fp, "%-9s\t%-22s", "itrj", "cv")

        for name in kinds_of_weights
            @printf(fp, "\t%-22s", "weight_$(name)")
        end

        @printf(fp, "\n")
    elseif (has_fp==true) && (p.kind_of_bias=="parametric")
        fp = open(p.measuredir * "/bias_data_$instance.txt", "w")
        kinds_of_weights = ["branduardi"]
        @printf(fp, "%-9s\t%-22s\t%-22s\n", "itrj", "cv", "weight_branduardi")
        @info ">> Parametric Bias is always STATIC"
        @info ">> Weight-type defaults to \"branduardi\", i.e. exp(V(Qᵢ)) on parametric bias"
        biasfile = nothing
    else
        biasfile = nothing
        kinds_of_weights = nothing
        fp = nothing
    end

    println_verbose1(verbose, "\t>> BIASFILE = $(biasfile)")
    write_bias_every = p.write_bias_every≡nothing ? 0 : p.write_bias_every
    println_verbose1(verbose, "\t>> WRITE_BIAS_EVERY = $(write_bias_every)")
    @assert (write_bias_every==0) || (write_bias_every>=p.stride)
    println_verbose1(verbose)

    # write to file after construction to make sure nothing went wrong
    biasfile≢nothing && write_to_file(bias, biasfile)

    return Bias(
        TCV(), smearing, is_static,
        bias, biasfile, write_bias_every,
        kinds_of_weights, fp,
    )
end

(b::Bias)(cv) = b.bias(cv)
kind_of_cv(b::Bias) = b.kind_of_cv
Base.close(b::Bias) = b.fp≢nothing ? close(b.fp) : nothing

include("metadynamics.jl")
include("opes.jl")
include("parametric.jl")

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

function in_bounds(cv, lb, ub)
    lb <= cv <= ub && return true
    return false
end

include("weights.jl")

end
