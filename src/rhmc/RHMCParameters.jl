# Adopted from https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/master/src/rhmc/rhmc.jl
module RHMCParameters

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
    RHMCParams(power::Rational, fun::Function; n_max=10, lambda_low=0.0004, lambda_high=64, tol=1e-6)

Return an `RHMCParams` which is a container for the Remez coefficients calculated with
the specified `power` on the interval `[lambda_low, lambda_high]` and the specified
`precision` (decimal digits). Orders `1:n_max` are tested after minimax refinement;
the first whose direct and inverse absolute errors both meet `tol` is returned.
The inverse is fitted and the direct approximation is its algebraic reciprocal.
If no order meets tolerance, the final candidate is returned with a warning.
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
        my_out = mpi_amroot() ? stdout : devnull
        coeffs, coeffs_inv, err, err_inv, n, n_inv = redirect_stderr(my_out) do
            calc_coefficients(num, den, n_max, lambda_low, lambda_high, precision, fun; tol)
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
    @assert n_max >= 1 "n_max must be positive"
    @assert 0 < lambda_low < lambda_high "invalid spectral interval"
    @assert tol > 0 "tol must be positive"
    # As in AlgRemez, precision specifies decimal digits. Keep all arithmetic,
    # including the residue calculation, at that precision until final conversion.
    return setprecision(BigFloat, ceil(Int, precision * log2(10))) do
        f(x) = fun(x)^(y//z)
        g(x) = fun(x)^(-y//z)
        itvl = interval(lambda_low, lambda_high, Base.precision(BigFloat))
        result = nothing
        for n in 1:n_max
            # AAA starts with two nodes (degree one). Disable its tolerance stop:
            # acceptance is decided only after Lawson and dense error refinement.
            r = approximate(g, itvl, AAA(); max_iter=n, tol=0, refinement=100,
                stagnation=n_max + 1)
            # At the iteration cap AAA may return an earlier, better interpolant.
            # Refine the requested order rather than silently testing it twice.
            r = minimax(RationalFunctionApproximation.rewind(r, length(r.history)), 20)
            inverse = r.fun
            # Swap barycentric numerator and denominator, preserving the stored
            # weighted values from Lawson rather than recomputing their product.
            direct = RationalFunctionApproximation.Barycentric(
                nodes(inverse), inv.(values(inverse)), inverse.w_times_f,
                weights(inverse))
            coeffs_inv = partial_fraction_coefficients(inverse)
            coeffs = partial_fraction_coefficients(direct)
            xs, _ = check(r; quiet=true, refinement=100)
            # Check the Float64 coefficients actually consumed by the solvers.
            # Include a logarithmic grid for resolution near the lower bound.
            append!(xs, exp.(range(log(BigFloat(lambda_low)), log(BigFloat(lambda_high));
                length=1001)))
            fitted = fittedfunction(coeffs)
            fitted_inv = fittedfunction(coeffs_inv)
            err = maximum(x -> abs(f(x) - fitted(x)), xs)
            err_inv = maximum(x -> abs(g(x) - fitted_inv(x)), xs)
            result = (coeffs, coeffs_inv, Float64(err), Float64(err_inv),
                coeffs.n, coeffs_inv.n)
            max(err, err_inv) <= tol && return result
        end
        @warn "RHMC approximation did not meet tolerance at n_max" n_max tol errors=result[3:4]
        return result
    end
end

# For a type (n,n) barycentric function, the constant at infinity is the
# ratio of the sums of numerator and denominator weights. No sample fit is needed.
function partial_fraction_coefficients(r)
    ps, rs = residues(r)
    all(x -> imag(x)<eps(Float64), ps) && all(x -> imag(x)<eps(Float64), rs) ||
        error("RHMC requires real poles and residues")
    all(p -> real(p) < 0, ps) || error("RHMC requires negative poles")
    order = sortperm(-real.(ps))
    α0 = Float64(sum(r.w_times_f) / sum(weights(r)))
    α = Tuple(Float64.(real.(rs[order])))
    β = Tuple(-Float64.(real.(ps[order])))
    return AlgRemezCoeffs(α0, α, β, length(ps))
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
