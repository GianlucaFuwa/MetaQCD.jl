# Adopted from https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/src/rhmc/rhmc.jl
module RHMCParameters

import ..AlgRemez: AlgRemezCoeffs, calc_coefficients

export RHMCParams
export get_n, get_α, get_α0, get_β, get_α_inverse, get_α0_inverse, get_β_inverse

"""
    RHMCParams(power::Rational; n=10, lambda_low=0.0004, lambda_high=64, precision=42)

Return an `RHMCParams` which is a container for the Remez coefficients calculated with
the specified `power` on the interval `[lambda_low, lamda_high]` and the specified
`precision`.
"""
struct RHMCParams{N}
    y::Int64
    z::Int64
    coeffs::AlgRemezCoeffs{N}
    coeffs_inverse::AlgRemezCoeffs{N}

    function RHMCParams(
        power::Rational; n=10, lambda_low=0.0004, lambda_high=64, precision=42
    )
        num = numerator(power)
        den = denominator(power)
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
        power = y//z
        num = numerator(power)
        den = denominator(power)
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
        return new{n}(num, den, coeffs, coeffs_inverse)
    end
end

get_n(::RHMCParams{N}) where {N} = N
get_α(x::RHMCParams) = x.coeffs.α
get_α0(x::RHMCParams) = x.coeffs.α0
get_β(x::RHMCParams) = x.coeffs.β
get_α_inverse(x::RHMCParams) = x.coeffs_inverse.α
get_α0_inverse(x::RHMCParams) = x.coeffs_inverse.α0
get_β_inverse(x::RHMCParams) = x.coeffs_inverse.β

end
