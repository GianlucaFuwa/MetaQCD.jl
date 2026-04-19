module MetaAnalysis

using ADerrors
using DelimitedFiles
using Dierckx
using LaTeXStrings
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
export Bootstrap, Jackknife, UWerr, analyze, modify_bias, plot, auto_correlation
export t0_scale, t0_w0_scale, t0_w0_scale_old

abstract type AbstractErrorEstimator end

const DEFAULT_COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00"]

include("measurements.jl")
include("bias.jl")
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

using Printf

function phys_not(val::Real, err::Real)
    if err == 0
        return string(val) * "(0)"
    end
    
    val, err = Float64(val), Float64(err)
    
    # Find the order of magnitude of the error
    err_order = floor(Int, log10(abs(err)))
    
    # The error determines the precision of our measurement
    # We'll round both value and error to this precision level
    precision_factor = 10.0^err_order
    
    # Round value and error to this precision
    val_rounded = round(val / precision_factor) * precision_factor
    err_rounded = round(err / precision_factor) * precision_factor
    
    # The error in parentheses is the error divided by the precision factor
    err_display = round(Int, err_rounded / precision_factor)
    
    # If the error display is >= 100, we need to shift our precision
    while err_display >= 100
        precision_factor *= 10
        err_order += 1
        val_rounded = round(val / precision_factor) * precision_factor
        err_rounded = round(err / precision_factor) * precision_factor
        err_display = round(Int, err_rounded / precision_factor)
    end
    
    # If error display is < 10, we might want 2 digits (like 32 from 0.0032)
    # This happens when we want to show more precision
    if err_display < 10 && err_order < 0
        # Try one more level of precision
        precision_factor /= 10
        err_order -= 1
        val_rounded = round(val / precision_factor) * precision_factor
        err_rounded = round(err / precision_factor) * precision_factor
        err_display = round(Int, err_rounded / precision_factor)
        
        # If this makes it too big, revert
        if err_display >= 100
            precision_factor *= 10
            err_order += 1
            val_rounded = round(val / precision_factor) * precision_factor
            err_rounded = round(err / precision_factor) * precision_factor
            err_display = round(Int, err_rounded / precision_factor)
        end
    end
    
    err_str = @sprintf("(%d)", err_display)
    
    # Determine if we need scientific notation
    abs_val = abs(val_rounded)
    use_scientific = false
    
    if abs_val > 0
        val_magnitude = floor(Int, log10(abs_val))
        # Use scientific notation for very large numbers or when precision is very small
        use_scientific = (val_magnitude >= 5) || (err_order <= -4)
    end
    
    if use_scientific && val_rounded != 0
        # Scientific notation
        exponent = floor(Int, log10(abs_val))
        mantissa = val_rounded / 10.0^exponent
        
        # For scientific notation, we need to recalculate the error representation
        # The error should be expressed in the same units as the mantissa
        mantissa_err = err / 10.0^exponent
        
        # Find appropriate precision for the mantissa based on mantissa_err
        if mantissa_err >= 1
            # Error is in the units place of mantissa
            mantissa_str = @sprintf("%.0f", round(mantissa))
            err_display_sci = round(Int, mantissa_err)
        else
            # Error is in decimal places of mantissa
            err_order_mantissa = floor(Int, log10(mantissa_err))
            precision_factor_mantissa = 10.0^err_order_mantissa
            
            mantissa_rounded = round(mantissa / precision_factor_mantissa) * precision_factor_mantissa
            err_display_sci = round(Int, mantissa_err / precision_factor_mantissa)
            
            # Handle case where we want 2 digits in error
            if err_display_sci < 10 && err_order_mantissa < 0
                precision_factor_mantissa /= 10
                mantissa_rounded = round(mantissa / precision_factor_mantissa) * precision_factor_mantissa
                err_display_sci = round(Int, mantissa_err / precision_factor_mantissa)
                if err_display_sci >= 100
                    precision_factor_mantissa *= 10
                    mantissa_rounded = round(mantissa / precision_factor_mantissa) * precision_factor_mantissa
                    err_display_sci = round(Int, mantissa_err / precision_factor_mantissa)
                end
            end
            
            decimal_places = max(0, -floor(Int, log10(abs(precision_factor_mantissa))))
            mantissa_str = @sprintf("%.*f", decimal_places, mantissa_rounded)
        end
        
        err_str_sci = @sprintf("(%d)", err_display_sci)
        return mantissa_str * err_str_sci * @sprintf("e%d", exponent)
    else
        # Regular notation
        if err_order >= 0
            # Integer precision
            val_str = @sprintf("%.0f", val_rounded)
        else
            # Decimal precision
            decimal_places = -err_order
            val_str = @sprintf("%.*f", decimal_places, val_rounded)
        end
        
        return val_str * err_str
    end
end

end
