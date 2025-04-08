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

function round_to_sigfigs(x::Float64, sigfigs::Int)
    if x == 0
        return 0.0
    end
    exponent = floor(Int, log10(abs(x)))
    factor = 10.0^(sigfigs - 1 - exponent)
    return round(x * factor) / factor
end

function physical_notation(value::Float64, uncertainty::Float64; sigfigs::Int=2)
    if uncertainty == 0
        return string(value)
    elseif isnan(value) || isnan(uncertainty)
        return "NaN"
    elseif !isfinite(value) || !isfinite(uncertainty)
        return "Inf"
    end

    # 1. Round uncertainty to significant digits
    unc_rounded = round_to_sigfigs(uncertainty, sigfigs)

    # 2. Determine decimal precision
    exponent = floor(Int, log10(unc_rounded))
    decimal_places = max(0, -exponent + sigfigs - 1)

    # 3. Round value
    val_rounded = round(value, digits=decimal_places)

    # 4. Format value
    fmt = Printf.Format("%.$(decimal_places)f")
    val_str = Printf.format(fmt, val_rounded)

    # 5. Decide how to show uncertainty
    if isinteger(unc_rounded * 10^decimal_places)
        # Integer-like uncertainty → use digits only
        unc_digits = Int(round(unc_rounded * 10^decimal_places))
        return "$(val_str)($(unc_digits))"
    else
        # Decimal-like uncertainty → format as-is
        fmt = Printf.Format("%.$(decimal_places)f")
        unc_str = Printf.format(fmt, unc_rounded)
        return "$(val_str)($(unc_str))"
    end
end

end
