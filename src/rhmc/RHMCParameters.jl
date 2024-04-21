# Adopted from https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/src/rhmc/rhmc.jl
module RHMCParameters

import ..AlgRemez: AlgRemezCoeffs, calc_coefficients

"""
    RHMCParams(order::Rational; n=10, lambda_low=0.0004, lambda_high=64, precision=42)

Return an `RHMCParams` which is a container for the Remez coefficients calculated with
the specified `order` on the interval `[lambda_low, lamda_high]` and the specified
`precision`.
"""
struct RHMCParams #type for the rational Hybrid Monte Carlo
    y::Int64
    z::Int64
    coeffs::AlgRemezCoeffs
    coeffs_inverse::AlgRemezCoeffs

    function RHMCParams(
        order::Rational; n=10, lambda_low=0.0004, lambda_high=64, precision=42
    )
        num = numerator(order)
        den = denominator(order)
        return RHMCParams(
            num,
            den;
            n=n,
            lambda_low=lambda_low,
            lambda_high=lambda_high,
            precision=precision,
        )
    end

    function RHMCParams(
        y::Int, z::Int; n::Int=10, lambda_low=4e-4, lambda_high=64, precision=42
    )
        order = y//z
        num = numerator(order)
        den = denominator(order)
        @assert num != 0 "numerator should not be zero!"
        @assert num * den != 1 "$(num ÷ den) should not be 1!"

        coeff_plus, coeff_minus = calc_coefficients(
            abs(num), den, n, lambda_low, lambda_high; precision=precision
        )

        if num > 0
            coeffs = coeff_plus
            coeffs_inverse = coeff_minus
        elseif num < 0
            coeffs_inverse = coeff_plus
            coeffs = coeff_minus
        end
        return new(num, den, coeffs, coeffs_inverse)
    end
end

function get_α(x::RHMCParams)
    return x.coeffs.α
end

function get_α0(x::RHMCParams)
    return x.coeffs.α0
end

function get_β(x::RHMCParams)
    return x.coeffs.β
end

function get_order(x::RHMCParams)
    return x.coeffs.n
end

function get_α_inverse(x::RHMCParams)
    return x.coeffs_inverse.α
end

function get_α0_inverse(x::RHMCParams)
    return x.coeffs_inverse.α0
end

function get_β_inverse(x::RHMCParams)
    return x.coeffs_inverse.β
end

end
