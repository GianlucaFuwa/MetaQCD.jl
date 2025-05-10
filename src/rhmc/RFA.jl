function calc_coefficients(
    f, ::Type{T}=Float64; a=0.0004, b=6.0, ndeg=10, npts=20
) where {T}
    r = approximate(f, myitvl(T); max_iter=ndeg);
    β = -Float64.(poles(r))

    # Sample xs — log-space for better small-x resolution
    xs = exp10.(range(log10(a), log10(b), length=npts))
    xs = filter(x -> all(abs(x + b) > 1e-10 for b in β), xs)

    # Evaluate r at sample points
    ys = Float64.(r.(xs))

    # Build A: [1, 1/(x + β1), ..., 1/(x + βn)]
    A = hcat(ones(length(xs)), [1 ./ (xs .+ β[j]) for j in 1:ndeg]...)

    # Solve with SVD
    U, S, V = svd(A)
    coeffs = V * Diagonal(1.0 ./ S) * (U' * ys)

    α0 = coeffs[1]
    α = coeffs[2:end]
    acoeffs = AlgRemezCoeffs(α0, tuple(α...), tuple(β...), ndeg)
    ff = fittedfunction(acoeffs)
    return ff
end
