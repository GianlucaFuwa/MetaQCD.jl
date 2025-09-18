module BiasModule

using DelimitedFiles
using LinearAlgebra
using Polyester: @batch
using StaticArrays
using StaticTools: StaticString
using Statistics
using ..MetaIO
using ..Parameters: ParameterSet
using ..Utils

import ..Fields: Gaugefield, WilsonGaugeAction, Plaquette, Clover
import ..Fields: SymanzikTreeGaugeAction, IwasakiGaugeAction, DBW2GaugeAction
import ..Fields: calc_gauge_action, gauge_action_deriv!, is_distributed
import ..Measurements: top_charge, top_charge_deriv!
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing, calc_smearedU!

abstract type AbstractBias end

# Convenience struct that contains the CV function and its derivative function
# `cv_temp_ind` is there to tell the `deriv_func` which kind of temporary field it needs
# for the calculation (e.g., a temporary `Tensorfield` or `Colorfield`)
struct CVinfo{F<:Function,dF<:Function,I}
    cv_func::F
    deriv_func::dF
    cv_temp_ind::I
end

struct NoBias end
(b::NoBias)(::Real) = 0.0
(b::NoBias)(::Any) = 0.0

# functions that have to be overloaded by each bias type
get_ext(::AbstractBias) = "" # Get file extension of the current bias
is_adaptive(::AbstractBias) = false # Check if the starting bin width can be adaptively set (only for OPES)
set_sigma0!(::AbstractBias) = nothing # Change the starting bin width of OPES bias
update!(::AbstractBias) = nothing # update the bias
write_to_file(::AbstractBias, args...) = nothing # write bias to file
ext_length(::AbstractBias) = Val(0) # Determine length of file extension statically

"""
    Bias{NumCV,BiasType,Smearing,Weights,BiasFile,DataFile}

Container for bias potential and metadata.

    Bias(p::ParameterSet, U::Gaugefield; instance=0, dummy=false, build=false)

Create a Bias that holds general parameters of bias enhanced sampling, like the kind of CV,
its smearing and filenames relevant to the bias. Also holds the specific kind
of bias (`Metadynamics`, `OPES` or `Parametric` for now).

The `instance` keyword is used in case of PT-MetaD and multiple walkers to assign the
correct `usebias` to each stream.

If `dummy=true` the bias is static and set to zero as for the measurement stream in PT-MetaD

If `build=true` certain things are made more convenient for the building of the bias, like
only the root rank printing its bias to file etc.
"""
mutable struct Bias{N,TB,TS,TW,T1,T2,T3}
    cv_numsmears::Vector{Int64}
    bias::TB
    smearing::TS
    kinds_of_weights::TW
    biasfile::T1
    datafile::T2
    buffers::T3
    CV::Vector{Float64}
    function Bias(
        cv_numsmears, bias::TB, smearing::TS, weights::TW, bfile::T1, dfile::T2, buffers::T3
    ) where {TB,TS,TW,T1,T2,T3}
        N = length(bias)
        CV = zeros(Float64, N)
        return new{N,TB,TS,TW,T1,T2,T3}(
            cv_numsmears, bias, smearing, weights, bfile, dfile, buffers, CV
        )
    end
end

function Bias(p, U; mpi_multi_sim=false, instance=mpi_myrank(), dummy=false, build=false)
    inum = if dummy
        0
    elseif mpi_multi_sim
        MPI_INSTANCE[]
    else
        instance
    end

    @level1("- Constructing Bias instance $(inum)...")

    rho = p.rhostout_for_cv
    biases = p.biases
    num_cv = length(biases)
    (num_cv == 0) && return NoBias()
    cv_numsmears = zeros(Int64, num_cv)

    bias = ntuple(num_cv) do i
        @level1("|")
        bias_parameters = bias_parameters_from_dict(biases[i], instance; build)
        name = bias_parameters.kind_of_cv
        numsmears = bias_parameters.numsmears_for_cv
        cv_numsmears[i] = numsmears
        @level1("|  Bias $i: $(bias_parameters.type)")
        @level1("|  CV$i: $(name) with $(numsmears)x$(rho) Stout smearing")
        if biases[i]["type"] ∈ ["metad", "metadynamics"]
            Metadynamics(
                bias_parameters;
                instance=instance, dummy=dummy, mpi_multi_sim=mpi_multi_sim, build=build
            )
        elseif biases[i]["type"] == "opes"
            OPES(
                bias_parameters;
                instance=instance, dummy=dummy, mpi_multi_sim=mpi_multi_sim, build=build
            )
        elseif biases[i]["type"] == "opesmt"
            OPESmultithermal(
                bias_parameters, p.beta;
                instance=instance, dummy=dummy, mpi_multi_sim=mpi_multi_sim, build=build
            )
        elseif biases[i]["type"] == "parametric"
            Parametric(bias_parameters; dummy=dummy)
        else
            error("type $(p[i]["type"]) not supported. Try metad, opes, opesmt or parametric")
        end
    end

    buffers = ntuple(length(bias)) do i
        create_buffer(bias[i])
    end

    smearing = StoutSmearing(U; numlayers=maximum(cv_numsmears), rho=rho)

    kinds_of_weights = if any(x -> !(x isa Metadynamics), bias)
        ["branduardi"]
    else
        p.weight_type
    end

    inum_str = lpad(inum, 3, "0")
    biasfile = ntuple(num_cv) do i
        ext = get_ext(bias[i])
        _name = joinpath(p.bias_dir, "bias$(i)_$(inum_str)$(ext)")

        # INFO: When using PT-MetaD, we want each instance to print its bias
        # When using multiple walkers during build, we only need instance 0 to print
        # since they are all the same anyway
        _biasfile = if mpi_amroot(mpi_comm_instance()) && !dummy && !build
            _name
        elseif mpi_amroot(mpi_comm_instance()) && !dummy && build
            mpi_amroot() ? _name : ""
        else
            ""
        end

        StaticString(_biasfile)
    end

    _datafile = joinpath(p.measure_dir, "bias_data_$(inum_str).txt")
    datafile = StaticString(_datafile)
    fp = fopen(_datafile, "w")
    printf(fp, "%-11s", "itrj")

    for i in 1:num_cv
        printf(fp, "%-25s", "cv$i")
    end

    for name in kinds_of_weights
        printf(fp, "%-25s", "weight_$(name)")
    end

    newline(fp)
    fclose(fp)

    @level1("|  BIASFILE: $(string(biasfile))")
    @level1("|  DATAFILE: $(string(datafile))")

    # write to file after construction to make sure nothing went wrong
    if mpi_amroot(mpi_comm_instance())
        for i in eachindex(bias)
            @level1("$(biasfile[i])")
            write_to_file(bias[i], biasfile[i])
        end
    end

    !isnothing(p.starting_Q) && @level1("|  STARTING SECTOR: $(string(p.starting_Q))")
    @level1("-")
    @level1("")
    return Bias(
        cv_numsmears,
        bias,
        smearing,
        kinds_of_weights,
        biasfile,
        datafile,
        buffers,
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

Base.length(::Bias{N}) where {N} = N
(b::Bias{N})(cv) where {N} = sum(b.bias[i](cv[i]) for i in 1:N)

set_cv!(bias::Bias, cv) = bias.CV .= cv
set_cv!(::NoBias, cv) = nothing
update_bias!(::NoBias, args...; kwargs...) = nothing
update_bias!(::Nothing, args...; kwargs...) = nothing
is_adaptive(b::Bias{N}) where {N} = ntuple(i -> is_adaptive(b.bias[i]), Val(N))
set_sigma0!(b::Bias, val, i) = set_sigma0!(b.bias[i], val)

include("bias_parameters.jl")
include("metadynamics.jl")
include("opes.jl")
include("opes_multithermal.jl")
include("parametric.jl")

function update_bias!(b::Bias{N}, itrj; mpi_multi_sim=false) where {N}
    return update_bias!(b, b.CV, itrj; mpi_multi_sim=mpi_multi_sim)
end

function update_bias!(
    b::Bias{N}, values, itrj; mpi_multi_sim=false
) where {N}
    (length(values) == 0) && return nothing

    for (icv, bias) in enumerate(b.bias)
        bias.static && continue
        values_i = ntuple(j -> values[j][icv], length(values))
        update!(bias, values_i, itrj)

        if (bias.write_bias_every != 0) && (itrj % bias.write_bias_every == 0)
            # INFO: When using parallel tempering with MPI, we only swap the bias
            # potentials and the "names" of the instances, i.e., the number associated to
            # the instances. Thus we have to change all filenames to conincide with the
            # current instance
            filename = if mpi_multi_sim
                set_ext!(b.biasfile[icv], ext_length(bias))
            else
                b.biasfile[icv]
            end

            if mpi_amroot(mpi_comm_instance())
                write_to_file(bias, filename)
            end
        end
    end

    return nothing
end

recalc_cv!(::Gaugefield, ::Nothing) = nothing
recalc_cv!(::Gaugefield, ::NoBias) = nothing

function recalc_cv!(U::Gaugefield, b::Bias{N}) where {N}
    CV_new = calc_cv(U, b)
    b.CV .= CV_new
    return nothing
end

function recalc_cv!(U::Vector{TG}, b::Vector{TB}) where {TG<:Gaugefield,TB<:Bias}
    for i in eachindex(U)
        recalc_cv!(U[i], b[i])
    end

    return nothing
end

calc_cv(U, ::Nothing, ::Bool=false) = 0.0
calc_cv(U, ::Nothing, ::Int64, ::Bool=false) = 0.0
calc_cv(U, ::NoBias, ::Bool=false) = 0.0
calc_cv(U, ::NoBias, ::Int64, ::Bool=false) = 0.0

function calc_cv(U, bias::AbstractBias)
    return bias.cvinfo.cv_func(U)
end

function calc_cv(U, b::Bias{N,TB,TS}, ::Bool=false) where {N,TB,TS<:NoSmearing} # all CVs
    return ntuple(i -> calc_cv(U, b.bias[i]), Val(N))
end

function calc_cv(U, b::Bias{N,TB,TS}, i::Int64, ::Bool=false) where {N,TB,TS<:NoSmearing} # 1 certain CV
    return calc_cv(U, b.bias[i])
end

function calc_cv(U, b::Bias{N}, is_smeared::Bool=false) where {N} # all CVs smeared
    is_smeared || calc_smearedU!(b.smearing, U)
    levels = b.cv_numsmears
    CV_new = ntuple(Val(N)) do i
        smeared_U = b.smearing.Usmeared_multi[levels[i]+1]
        calc_cv(smeared_U, b.bias[i])
    end

    return CV_new
end

function calc_cv(U, b::Bias{N}, i::Int64, is_smeared::Bool=false) where {N} # 1 certain CV smeared
    is_smeared || calc_smearedU!(b.smearing, U)
    levels = b.cv_numsmears
    smeared_U = b.smearing.Usmeared_multi[levels[i]+1]
    return calc_cv(smeared_U, b.bias[i])
end

calc_cv_deriv!(dU, b::Bias, i, args...) = b.bias[i].cvinfo.deriv_func(dU, args...)

∂V∂Q(b::NoBias, ::Any) = 0.0
∂V∂Q(b::Bias, cv::Float64, i) = ∂V∂Q(b.bias[i], cv)
∂V∂Q(b::Bias, cv, i) = ∂V∂Q(b.bias[i], cv[i])

function in_bounds(cv, lb, ub)
    lb <= cv < ub && return true
    return false
end

include("weights.jl")

# In order to write and load the bias easily with JLD2 for checkpointing, we need to define
# custom serialization, because saving and loading IOStreams doesn't work
using JLD2

struct BiasSerialization{N,TB,TS,TW,T1,T2}
    cv_numsmears::Vector{Int64}
    bias::TB
    smearing::TS
    kinds_of_weights::TW
    biasfile::T1
    datafile::T2
end

function JLD2.writeas(::Type{<:Bias{CV,TS,TB,TW}}) where {CV,TS,TB,TW}
    return BiasSerialization{CV,TS,TB,TW}
end

function Base.convert(::Type{<:BiasSerialization}, b::Bias)
    out = BiasSerialization(
        b.cv_numsmears,
        b.bias,
        b.smearing,
        b.kinds_of_weights,
        b.biasfile,
        b.datafile,
    )
    return out
end

function Base.convert(::Type{<:Bias}, b::BiasSerialization)
    fp = open(b.datafile, "a")
    out = Bias(
        b.cv_numsmears,
        b.bias,
        b.smearing,
        b.kinds_of_weights,
        b.biasfile,
        b.datafile,
        fp,
    )
    return out
end

end
