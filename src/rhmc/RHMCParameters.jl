# Adopted from https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/src/rhmc/rhmc.jl
module RHMCParameters

using LinearAlgebra
using RationalFunctionApproximation
using ..Utils

# import ..AlgRemez: AlgRemezCoeffs, calc_coefficients

export RHMCParams
export get_n, get_n_inverse, get_α, get_α0, get_β, get_α_inverse
export get_α0_inverse, get_β_inverse

struct AlgRemezCoeffs{N}
    α0::Float64
    α::NTuple{N,Float64}
    β::NTuple{N,Float64}
    n::Int64
end

function Base.display(x::AlgRemezCoeffs)
    println("""
        f(x) = α0 + sum_i^n α[i]/(x + β[i])
    """)
    println("Order: $(x.n)")
    println("α0: $(x.α0)")
    println("α: $(x.α)")
    return println("β: $(x.β)")
end

function fittedfunction(coeff)
    function func(x)
        value = coeff.α0
        for i in 1:coeff.n
            value += coeff.α[i] / (x + coeff.β[i])
        end
        return value
    end
    return x -> func(x)
end

"""
    RHMCParams(power::Rational, fun::Function; n=10, lambda_low=0.0004, lambda_high=64, tol=1e-6)

Return an `RHMCParams` which is a container for the Remez coefficients calculated with
the specified `power` on the interval `[lambda_low, lambda_high]` and the specified
`precision`.
"""
struct RHMCParams{N1,N2}
    coeffs::AlgRemezCoeffs{N1}
    coeffs_inverse::AlgRemezCoeffs{N2}
    lambda_low::Float64
    lambda_high::Float64
    maxerr::NTuple{2,Float64}
    y::Int64
    z::Int64
    function RHMCParams(
        power, fun=x -> x;
        n_max=10, lambda_low=0.0004, lambda_high=64, precision=42, tol=1000eps(Float64)
    )
        num = numerator(power)
        den = denominator(power)
        @assert den != 0 "denominator should not be zero!"
        @assert num != 0 "numerator should not be zero!"
        @assert num * den != 1 "power should not be 1!"

        # suppress warnings from RationalFunctionApproximation.jl here
        my_out = mpi_amroot() ? stdout : stdout
        coeffs, coeffs_inv, err, err_inv, n, n_inv = redirect_stderr(my_out) do
            calc_coefficients(num, den, n_max+1, lambda_low, lambda_high, precision, fun; tol)
        end
        maxerr = (err, err_inv)
        return new{n,n_inv}(coeffs, coeffs_inv, lambda_low, lambda_high, maxerr, num, den)
    end
end

get_n(::RHMCParams{N1,N2}) where {N1,N2} = N1
get_n_inverse(::RHMCParams{N1,N2}) where {N1,N2} = N2
get_α(x::RHMCParams) = x.coeffs.α
get_α0(x::RHMCParams) = x.coeffs.α0
get_β(x::RHMCParams) = x.coeffs.β
get_α_inverse(x::RHMCParams) = x.coeffs_inverse.α
get_α0_inverse(x::RHMCParams) = x.coeffs_inverse.α0
get_β_inverse(x::RHMCParams) = x.coeffs_inverse.β

function calc_coefficients(
    y, z, n_max, lambda_low, lambda_high, precision, fun::Function=x -> x; tol=1000 * eps(Float64)
)
    @assert y > 0 && z > 0 "Inputs y and z need to be positive"
    f(x) = fun(x)^(y//z)
    g(x) = fun(x)^(-y//z)
    itvl = interval(lambda_low, lambda_high, precision)

    r_p = approximate(f, itvl; max_iter=n_max, tol=tol, refinement=100, stagnation=100)
    # r_p = minimax(r_p, 20)
    err_p = maximum(check(r_p, quiet=true, refinement=100)[2])
    n_p = degree(r_p)

    r_m = approximate(g, itvl; max_iter=n_max, tol=tol, refinement=100, stagnation=100)
    # r_m = minimax(r_m, 20)
    err_m = maximum(check(r_m, quiet=true, refinement=100)[2])
    n_m = degree(r_m)

    β_p = -Float64.(poles(r_p))
    β_m = -Float64.(poles(r_m))

    # Sample xs — log-space for better small-x resolution
    xs_p = exp10.(range(log10(lambda_low), log10(lambda_high); length=100))
    xs_p = filter(x -> all(abs(x + b) > 1e-10 for b in β_p), xs_p)
    xs_m = exp10.(range(log10(lambda_low), log10(lambda_high); length=100))
    xs_m = filter(x -> all(abs(x + b) > 1e-10 for b in β_m), xs_m)

    # Evaluate r at sample points
    ys_p = Float64.(r_p.(xs_p))
    ys_m = Float64.(r_m.(xs_m))

    # Build A: [1, 1/(x + β1), ..., 1/(x + βn)]
    A_p = hcat(ones(length(xs_p)), [1 ./ (xs_p .+ β_p[j]) for j in 1:n_p]...)
    A_m = hcat(ones(length(xs_m)), [1 ./ (xs_m .+ β_m[j]) for j in 1:n_m]...)

    # Solve with SVD
    U_p, S_p, V_p = svd(A_p)
    U_m, S_m, V_m = svd(A_m)
    coeffs_p = V_p * Diagonal(1.0 ./ S_p) * (U_p' * ys_p)
    coeffs_m = V_m * Diagonal(1.0 ./ S_m) * (U_m' * ys_m)

    α0_p = coeffs_p[1]
    α_p = coeffs_p[2:end]
    acoeffs_p = AlgRemezCoeffs(α0_p, tuple(α_p...), tuple(β_p...), n_p)

    α0_m = coeffs_m[1]
    α_m = coeffs_m[2:end]
    acoeffs_m = AlgRemezCoeffs(α0_m, tuple(α_m...), tuple(β_m...), n_m)
    return acoeffs_p, acoeffs_m, err_p, err_m, n_p, n_m
end

@inline function interval(a, b, precision=42)
    return RationalFunctionApproximation.Segment(BigFloat(a; precision), BigFloat(b; precision))
    # return RationalFunctionApproximation.Segment(Float64(a), Float64(b))
end

function Base.show(io::IO, ::MIME"text/plain", rhmc::RHMCParams{N1,N2}) where {N1,N2}
    print(
        io,
        "Order: $(N1), $(N2) (inverse), ",
        "Range: [$(rhmc.lambda_low), $(rhmc.lambda_high)], ",
        "Maxerr: $(rhmc.maxerr[1]), $(rhmc.maxerr[2]) (inverse)",
    )
    return nothing
end

function Base.show(io::IO, rhmc::RHMCParams{N1,N2}) where {N1,N2}
    print(
        io,
        "Order: $(N1), $(N2) (inverse), ",
        "Range: [$(rhmc.lambda_low), $(rhmc.lambda_high)], ",
        "Maxerr: $(rhmc.maxerr[1]), $(rhmc.maxerr[2]) (inverse)",
    )
    return nothing
end

end
