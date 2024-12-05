module MetaAnalysis

using ADerrors
using DelimitedFiles
using Dierckx
using LsqFit
using Plots
using Polyester
using Printf
using ProgressMeter
using RecipesBase
using Roots
using SingularSpectrumAnalysis
using Statistics

export MetaMeasurements, MetaBias, eigenvalues, hadroncorrelator, timeseries, observables
export Bootstrap, Jackknife, UWerr, analyze, modify_bias, plot
export t0_scale

abstract type AbstractErrorEstimator end

const DEFAULT_COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00"]

include("bias.jl")
include("measurements.jl")
include("autocorr.jl")
include("bootstrap.jl")
include("jackknife.jl")
include("uwerr.jl")
include("scale_t0_w0.jl")
include("biasmod.jl")

function analyze(
    filename::String,
    error_est::AbstractErrorEstimator;
    weight_filename = nothing,
    func = identity,
    itrj_range = nothing,
    save_filename = nothing,
    str = "",
    column = 1,
    skipstart = 0,
    comments = true,
    comment_char = '#',
)
    println("\n| Begin analysis...\n")
    data, headers = readdlm(
        filename,
        Float64,
        comments = comments,
        comment_char = comment_char,
        header = true,
    )

    @assert column + 1 ∈ axes(data, 2) "specified column number not in bounds"

    if weight_filename === nothing
        weights = nothing
    else
        weights = readdlm(weight_filename, Float64, comments=true)[1+skipstart:end, 2]
    end

    results = analyze(
        data[1+skipstart:end, column + 1],
        error_est,
        headers[column + 1],
        weights = weights,
        func = func,
        itrj_range = itrj_range,
        save_filename = save_filename,
        str = str,
    )

    display(results)

    return headers, data, weights, results
end

function analyze(
    data::AbstractArray{<:Real,1},
    error_est::AbstractErrorEstimator,
    header = "";
    weights = nothing,
    func = identity,
    itrj_range = nothing,
    save_filename = nothing,
    str = "",
)
    data_range = itrj_range===nothing ? axes(data, 1) : itrj_range

    datamean, dataerr, τ = error_est(
        func.(data[data_range]),
        weights,
    )

    if save_filename !== nothing && save_filename != ""
        fp = open(save_filename, "a+")
        head =
            "### $str ###\n" *
            "$(rpad("# mean", 17, " "))\t" *
            "$(rpad("stderr", 17, " "))\t" *
            "$(rpad("tauint", 17, " "))"
        println(fp, head)
        datamean_s = @sprintf("%.15E", datamean)
        dataerr_s = @sprintf("%.15E", dataerr)
        τ_s = @sprintf("%.15E", τ)
        println(fp, "$(datamean_s)\t$(dataerr_s)\t$τ_s\n")
        close(fp)
    end

    header = str !== "" ? str : header
    results = Dict(
        "header" => header,
        "mean" => datamean,
        "stderr" => dataerr,
        "τ" => τ,
    )
    return results
end

phys_not(val::uwreal) = phys_not(value(val), ADerrors.err(val))

function phys_not(val::Float64, err::Float64)
    err > 1 && return phys_not_cp(val, err)
    exp_err = round(Int64, log10(err), RoundDown)-1
    err_shifted = err / 10.0^exp_err
    val_str = if exp_err < -5
        @sprintf("%f", round(val, digits=-exp_err))
    else
        @sprintf("%g", round(val, digits=-exp_err))
    end
    xx = abs(val) > 1 ? Int(2 + floor(log10(abs(val)))) : 2
    val_str = length(val_str)!=-exp_err+xx ? rpad(val_str, -exp_err+xx, "0") : val_str
    err_str = "($(round(Int64, err_shifted)))"
    return val_str * err_str
end

function phys_not(val::Int64, err::Int64)
    exp_err = round(Int64, log10(err), RoundDown)
    err_shifted = err / 10^exp_err
    val_str = @sprintf("%g", round(val, digits=-exp_err))
    val_str = length(val_str)!=-exp_err+2 ? rpad(val_str, -exp_err+2, "0") : val_str
    err_str = "($(round(Int64, err_shifted)))"
    return val_str * err_str
end

function phys_not_cp(val::Float64, err::Float64)
    # Calculate the exponent of the error
    exp_err = floor(log10(err))

    # Shift the error to have 2 significant digits
    err_shifted = exp_err > 1 ? round(err / 10^exp_err, sigdigits=2) : round(err, sigdigits=2)
    num_nachkomma = length(strip(splitext("$err_shifted")[2], '.'))

    # Format the value without trailing zeroes
    val_str = strip(@sprintf("%.2f", round(val, digits=num_nachkomma)), '0')
    val_str = strip(val_str, '.')

    # Format the error with trailing zeroes
    err_str = @sprintf("%.2f", err_shifted)
    err_str = rstrip(err_str, '.')
    err_str = rstrip(err_str, '0')
    err_str = rstrip(err_str, '.')
    err_str = @sprintf("(%s)", err_str)

    # Return the formatted string
    return val_str * err_str
end

end
