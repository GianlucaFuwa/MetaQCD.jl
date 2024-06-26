module BiasModule

using Base.Threads
using DelimitedFiles
using MPI
using Printf
using Statistics
using Unicode
using ..Parameters: ParameterSet
using ..Output

import ..Gaugefields: Gaugefield, Plaquette, Clover
import ..Measurements: top_charge
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing, calc_smearedU!

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const myrank = MPI.Comm_rank(comm)
const comm_size = MPI.Comm_size(comm)

abstract type AbstractBias end

"""
    Bias(p::ParameterSet, U::Gaugefield; instance=1)

Container that holds general parameters of bias enhanced sampling, like the kind of CV,
its smearing and filenames/-pointers relevant to the bias. Also holds the specific kind
of bias (`Metadynamics`, `OPES` or `Parametric` for now). \\
The `instance` keyword is used in case of PT-MetaD and multiple walkers to assign the
correct `usebias` to each stream. `has_fp` indicates whether the stream prints to file
at any point, since only rank 0 should print in case of MPI usage.
"""
struct Bias{TCV,TS,TB,T}
    kind_of_cv::TCV
    smearing::TS
    is_static::Bool
    bias::TB
    biasfile::String
    write_bias_every::Int64
    kinds_of_weights::Vector{String}
    fp::T
end

function Bias(p::ParameterSet, U; instance=1)
    @level1("┌ Setting Bias instance $(instance)...")
    kind_of_bias = Unicode.normalize(p.kind_of_bias, casefold=true)
    TCV = get_cvtype_from_parameters(p)
    smearing = StoutSmearing(U, p.numsmears_for_cv, p.rhostout_for_cv)
    is_static = instance==0 ? true : p.is_static[instance]
    sstr = (is_static || kind_of_bias=="parametric") ? "static" : "dynamic"
    @level1("|  Type: $(sstr) $(kind_of_bias)")

    if kind_of_bias ∈ ["metad", "metadynamics"]
        bias = Metadynamics(p; instance=instance)
    elseif kind_of_bias == "opes"
        bias = OPES(p; instance=instance)
    elseif kind_of_bias == "parametric"
        bias = Parametric(p; instance=instance)
    else
        error("kind_of_bias $(kind_of_bias) not supported. Try metad, opes or parametric")
    end

    @level1("|  CV: $TCV with $(p.numsmears_for_cv) x $(p.rhostout_for_cv) Stout")

    if myrank==0 && !(bias isa Parametric)
        is_opes = bias isa opes
        ext = is_opes ? "opes" : "metad"
        biasfile = p.biasdir * "/stream_$(instance).$(ext)"
        fp = open(p.measuredir * "/bias_data_$instance.txt", "w")
        kinds_of_weights = is_opes ? ["opes"] : p.kinds_of_weights
        str = @sprintf("%-9s\t%-22s", "itrj", "cv")
        print(fp, str)

        for name in kinds_of_weights
            str = @sprintf("\t%-22s", "weight_$(name)")
            print(fp, str)
        end

        println(fp)
    elseif myrank==0 && (bias isa Parametric)
        fp = open(p.measuredir * "/bias_data_$instance.txt", "w")
        kinds_of_weights = ["branduardi"]
        str = @sprintf("%-9s\t%-22s\t%-22s\n", "itrj", "cv", "weight_branduardi")
        println(fp, str)
        @level1("|  @info: Parametric bias defaults to static and weight-type \"branduardi\"")
        biasfile = ""
    else
        biasfile = ""
        kinds_of_weights = nothing
        fp = nothing
    end

    @level1("|  BIASFILE: $(biasfile)")
    write_bias_every = p.write_bias_every
    @level1("|  WRITE_BIAS_EVERY: $(write_bias_every)")
    @assert (write_bias_every==0) || (write_bias_every>=p.stride)

    # write to file after construction to make sure nothing went wrong
    write_to_file(bias, biasfile)
    @level1("└\n")
    return Bias(TCV(), smearing, is_static,
                bias, biasfile, write_bias_every,
                kinds_of_weights, fp)
end

function Base.show(io::IO, b::Bias)
	print(io, "$(typeof(b))", "(;")
    for fieldname in fieldnames(typeof(b))
		if fieldname == :smearing
        	print(io, " ", fieldname, " = ", typeof(getfield(b, fieldname)), ",")
		else
			print(io, " ", fieldname, " = ", getfield(b, fieldname), ",")
		end
    end
    print(io, ")")
	return nothing
end

(b::Bias)(cv) = b.bias(cv)
kind_of_cv(b::Bias) = b.kind_of_cv
Base.close(b::Bias{TCV,TS,TB,T}) where {TCV,TS,TB,T} = T≢Nothing ? close(b.fp) : nothing
update_bias!(::Nothing, args...) = nothing
write_to_file(::AbstractBias, args...) = nothing

include("metadynamics.jl")
include("opes.jl")
include("parametric.jl")

function update_bias!(b::Bias, values, itrj, write)
    b.is_static && return nothing
    update!(b.bias, values, itrj)
    write && write_to_file(b.bias, b.biasfile)
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

function calc_CV(U, ::Bias{TCV,TS}) where {TCV,TS<:NoSmearing}
    return top_charge(TCV(), U)
end

function calc_CV(U, b::Bias{TCV}) where {TCV}
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
    lb <= cv < ub && return true
    return false
end

include("weights.jl")

end
