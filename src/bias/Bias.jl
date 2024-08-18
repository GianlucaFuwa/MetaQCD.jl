module BiasModule

using Base.Threads
using DelimitedFiles
using MPI
using Printf
using Statistics
using Unicode
using ..Parameters: ParameterSet
using ..Output

import ..Fields: GaugeField, Plaquette, Clover
import ..Measurements: top_charge
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing, calc_smearedU!

const COMM = MPI.COMM_WORLD
const MYRANK = MPI.Comm_rank(COMM)
const COMM_SIZE = MPI.Comm_size(COMM)

abstract type AbstractBias end

struct NoBias end
(b::NoBias)(::Real) = 0.0

"""
    Bias{TopChargeType,Smearing,BiasType,Weights,FileType}
    
Container for bias potential and metadata.

    Bias(p::ParameterSet, U::GaugeField; instance=1)

Create a Bias that holds general parameters of bias enhanced sampling, like the kind of CV,
its smearing and filenames relevant to the bias. Also holds the specific kind
of bias (`Metadynamics`, `OPES` or `Parametric` for now).

The `instance` keyword is used in case of PT-MetaD and multiple walkers to assign the
correct `usebias` to each stream.
"""
struct Bias{TCV,TS,TB,TW,T}
    kind_of_cv::TCV
    smearing::TS
    is_static::Bool
    bias::TB
    kinds_of_weights::TW
    biasfile::T
    datafile::T
    write_bias_every::Int64
end

function Bias(p::ParameterSet, U; use_mpi=false, instance=1)
    @level1("┌ Setting Bias instance $(instance)...")
    kind_of_bias = Unicode.normalize(p.kind_of_bias; casefold=true)
    TCV = get_cvtype_from_parameters(p)
    smearing = StoutSmearing(U, p.numsmears_for_cv, p.rhostout_for_cv)
    is_static = instance == 0 ? true : p.is_static[instance]
    sstr = (is_static || kind_of_bias == "parametric") ? "static" : "dynamic"
    inum = use_mpi ? MYRANK+1 : instance
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

    @level1("|  CV: $TCV with $(smearing)")

    if !(bias isa Parametric)
        is_opes = bias isa OPES
        kinds_of_weights = is_opes ? ["opes"] : p.kinds_of_weights
        ext = is_opes ? "opes" : "metad"
        biasfile = MYRANK==0 ? joinpath(p.bias_dir, "stream_$(inum).$(ext)") : ""
        datafile = joinpath(p.measure_dir, "bias_data_$inum.txt")
        # FIXME: For some reason this errors with MPI on the UNI's cluster
        open(datafile, "w") do fp
            @printf(fp, "%-11s%-25s", "itrj", "cv")

            for name in kinds_of_weights
                @printf(fp, "%-25s", "weight_$(name)")
            end
            cnewline(fp)
        end
    elseif bias isa Parametric
        kinds_of_weights = ["branduardi"]
        biasfile = ""
        datafile = joinpath(p.measure_dir, "bias_data_$inum.txt")
        open(datafile, "w") do fp
            @printf(fp, "%-11s%-25s%-25s", "itrj", "cv", "weight_branduardi")
            cnewline(fp)
        end
        @level1(
            "|  @info: Parametric bias defaults to static and weight-type \"branduardi\""
        )
    end

    @level1("|  BIASFILE: $(biasfile)")
    write_bias_every = p.write_bias_every
    if write_bias_every <= p.stride
        write_bias_every = p.stride
    end
    @level1("|  WRITE_BIAS_EVERY: $(write_bias_every)")
    @assert write_bias_every == 0

    # write to file after construction to make sure nothing went wrong
    MYRANK == 0 && write_to_file(bias, biasfile)
    @level1("└")
    @level1("")
    return Bias(
        TCV(),
        smearing,
        is_static,
        bias,
        kinds_of_weights,
        biasfile,
        datafile,
        write_bias_every,
    )
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

kind_of_cv(::NoBias) = nothing
kind_of_cv(b::Bias) = b.kind_of_cv
update_bias!(::NoBias, args...) = nothing
update_bias!(::Nothing, args...) = nothing
write_to_file(::AbstractBias, args...) = nothing

include("metadynamics.jl")
include("opes.jl")
include("parametric.jl")

function update_bias!(b::Bias, values, itrj)
    (b.is_static || length(values) == 0) && return nothing
    update!(b.bias, values, itrj)
    if itrj % b.write_bias_every == 0
        (MYRANK == 0) && write_to_file(b.bias, b.biasfile)
    end
    return nothing
end

recalc_CV!(::GaugeField, ::Nothing) = nothing
recalc_CV!(::GaugeField, ::NoBias) = nothing

function recalc_CV!(U::GaugeField, b::Bias)
    CV_new = calc_CV(U, b)
    U.CV = CV_new
    return nothing
end

function recalc_CV!(U::Vector{TG}, b::Vector{TB}) where {TG<:GaugeField,TB<:Bias}
    for i in eachindex(U)
        recalc_CV!(U[i], b[i])
    end
    return nothing
end

calc_CV(U, ::Nothing, ::Bool=false) = U.CV
calc_CV(U, ::NoBias, ::Bool=false) = U.CV

function calc_CV(U, ::Bias{TCV,TS}, ::Bool=false) where {TCV,TS<:NoSmearing}
    return top_charge(TCV(), U)
end

function calc_CV(U, b::Bias{TCV}, is_smeared=false) where {TCV}
    is_smeared || calc_smearedU!(b.smearing, U)
    fully_smeared_U = b.smearing.Usmeared_multi[end]
    CV_new = top_charge(TCV(), fully_smeared_U)
    return CV_new
end

∂V∂Q(b::NoBias, ::Real) = 0.0
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

# In order to write and load the bias easily with JLD2 for checkpointing, we need to define
# custom serialization, because saving and loading IOStreams doesn't work
using JLD2

struct BiasSerialization{TCV,TS,TB,TW}
    kind_of_cv::TCV
    smearing::TS
    is_static::Bool
    bias::TB
    kinds_of_weights::TW
    biasfile::String
    datafile::String
    write_bias_every::Int64
end

function JLD2.writeas(::Type{<:Bias{TCV,TS,TB,TW}}) where {TCV,TS,TB,TW}
    return BiasSerialization{TCV,TS,TB,TW}
end

function Base.convert(::Type{<:BiasSerialization}, b::Bias)
    out = BiasSerialization(
        b.kind_of_cv,
        b.smearing,
        b.is_static,
        b.bias,
        b.kinds_of_weights,
        b.biasfile,
        b.datafile,
        b.write_bias_every,
    )
    return out
end

function Base.convert(::Type{<:Bias}, b::BiasSerialization)
    fp = open(b.datafile, "a")
    out = Bias(
        b.kind_of_cv,
        b.smearing,
        b.is_static,
        b.bias,
        b.kinds_of_weights,
        b.biasfile,
        b.datafile,
        b.write_bias_every,
        fp,
    )
    return out
end

end
