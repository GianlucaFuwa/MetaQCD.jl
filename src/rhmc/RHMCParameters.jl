# Adopted from https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/src/rhmc/rhmc.jl
module RHMCParameters

using LinearAlgebra
using RationalFunctionApproximation

import ..AlgRemez: AlgRemezCoeffs, calc_coefficients

export RHMCParams
export get_n, get_α, get_α0, get_β, get_α_inverse, get_α0_inverse, get_β_inverse

"""
    RHMCParams(power::Rational; n=10, lambda_low=0.0004, lambda_high=64, precision=42)

Return an `RHMCParams` which is a container for the Remez coefficients calculated with
the specified `power` on the interval `[lambda_low, lambda_high]` and the specified
`precision`.
"""
struct RHMCParams{N}
    coeffs::AlgRemezCoeffs{N}
    coeffs_inverse::AlgRemezCoeffs{N}
    lambda_low::Float64
    lambda_high::Float64
    maxerr::NTuple{2,Float64}
    y::Int64
    z::Int64
    function RHMCParams(
        y::Int, z::Int; n::Int=10, lambda_low=4e-4, lambda_high=64, precision=42
    )
        power = y//z
        num = numerator(power)
        den = denominator(power)
        @assert num != 0 "numerator should not be zero!"
        @assert num * den != 1 "$(num ÷ den) should not be 1!"

        # coeff_plus, coeff_minus = calc_coefficients(
        #     abs(num), den, n, lambda_low, lambda_high; precision=precision
        # )
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
        return new{n}(coeffs, coeffs_inverse, lambda_low, lambda_high, precision, num, den)
    end

    function RHMCParams(
        power::Rational, fun::Function=x->x; n=10, lambda_low=0.0004, lambda_high=64
    )
        num = numerator(power)
        den = denominator(power)
        @assert den != 0 "denominator should not be zero!"
        @assert num != 0 "numerator should not be zero!"
        @assert num * den != 1 "power should not be 1!"
        coeffs, coeffs_inverse, err, err_inverse = calc_coefficients(
            y, z, n, lambda_low, lambda_high, fun
        )
        maxerr = (err, err_inverse)
        return new{n}(coeffs, coeffs_inverse, lambda_low, lambda_high, maxerr, num, den)
    end

    # TODO: RationalFunctionApproximations for functions not strictly of type x^α
    # function RHMCParams(
    #     fun::Function; max_degree=20, lambda_low=0.0004, lambda_high=64,0, tol=1000*eps(Float64)
    # )
    #     num = numerator(power)
    #     den = denominator(power)
    #     return RHMCParams(
    #         num,
    #         den;
    #         n=n,
    #         lambda_low=lambda_low,
    #         lambda_high=lambda_high,
    #         precision=precision,
    #     )
    # end
end

function Base.show(io::IO, ::MIME"text/plain", rhmc::RHMCParams{N}) where {N}
    print(
        io,
        "ORDER: $N, SPECTRAL RANGE: [$(rhmc.lambda_low), $(rhmc.lambda_high)], PREC: $(rhmc.precision))"
    )
    return nothing
end

function Base.show(io::IO, rhmc::RHMCParams{N}) where {N}
    print(
        io,
        "ORDER: $N, SPECTRAL RANGE: [$(rhmc.lambda_low), $(rhmc.lambda_high)], PREC: $(rhmc.precision))"
    )
    return nothing
end

get_n(::RHMCParams{N}) where {N} = N
get_α(x::RHMCParams) = x.coeffs.α
get_α0(x::RHMCParams) = x.coeffs.α0
get_β(x::RHMCParams) = x.coeffs.β
get_α_inverse(x::RHMCParams) = x.coeffs_inverse.α
get_α0_inverse(x::RHMCParams) = x.coeffs_inverse.α0
get_β_inverse(x::RHMCParams) = x.coeffs_inverse.β

function calc_coefficients(y, z, n, lambda_low, lambda_high, fun::Function=x->x)
    @assert y > 0 && z > 0 "Inputs y and z need to be positive"
    f(x) = fun(x)^(y//z)
    g(x) = 1 / f(x)
    r_p = approximate(f, interval(lambda_low, lambda_high, Float64); max_iter=n);
    err_p = maximum(check(r_p)[2])
    r_m = approximate(g, interval(lambda_low, lambda_high, Float64); max_iter=n);
    err_m = maximum(check(r_m)[2])
    β_p = -Float64.(poles(r_p))
    β_m = -Float64.(poles(r_m))

    # Sample xs — log-space for better small-x resolution
    xs_p = exp10.(range(log10(lambda_low), log10(lambda_high), length=100))
    xs_p = filter(x -> all(abs(x + b) > 1e-10 for b in β_p), xs_p)
    xs_m = exp10.(range(log10(lambda_low), log10(lambda_high), length=100))
    xs_m = filter(x -> all(abs(x + b) > 1e-10 for b in β_m), xs_m)

    # Evaluate r at sample points
    ys_p = Float64.(r_p.(xs_p))
    ys_m = Float64.(r_m.(xs_m))

    # Build A: [1, 1/(x + β1), ..., 1/(x + βn)]
    A_p = hcat(ones(length(xs_p)), [1 ./ (xs_p .+ β_p[j]) for j in 1:n]...)
    A_m = hcat(ones(length(xs_m)), [1 ./ (xs_m .+ β_m[j]) for j in 1:n]...)

    # Solve with SVD
    U_p, S_p, V_p = svd(A_p)
    U_m, S_m, V_m = svd(A_m)
    coeffs_p = V_p * Diagonal(1.0 ./ S_p) * (U_p' * ys_p)
    coeffs_m = V_m * Diagonal(1.0 ./ S_m) * (U_m' * ys_m)

    α0_p = coeffs_p[1]
    α_p = coeffs_p[2:end]
    acoeffs_p = AlgRemezCoeffs(α0_p, tuple(α_p...), tuple(β_p...), n)

    α0_m = coeffs_m[1]
    α_m = coeffs_m[2:end]
    acoeffs_m = AlgRemezCoeffs(α0_m, tuple(α_m...), tuple(β_m...), n)
    return acoeffs_p, acoeffs_m, err_p, err_m
end

@inline function interval(a, b, ::Type{T}=Float64) where {T}
    return RationalFunctionApproximation.Segment(a, b)
end

end
