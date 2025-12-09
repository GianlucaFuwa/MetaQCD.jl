function cg_mixed!(
    x_high, A_high, A_low, b_high, r_high, Ap_high,
    x_low, r_low, Ap_low, p_low, r_old_low;
    delta=0.1, tol=1e-7, maxiters=1000, datafile=""
)
    rel_tol = tol * sqrt(real(dot(b_high, b_high)))
    mul!(Ap_high, A_high, x_high)
    copy!(r_high, b_high)
    axpy!(-1, Ap_high, r_high)
    res_high = real(dot(r_high, r_high))
    res_high_old = res_high
    res_low = res_high

    if sqrt(res_high) < rel_tol
        @level3 "|  Mixed CG: converged at iter 0 with res = $(sqrt(res_high))"
        print_solverdata(datafile, 0, sqrt(res_high))
        return 0, sqrt(res_high)
    end

    @level4 "|  Mixed CG: residual 0 = $(sqrt(res_high))"

    copy!(p_low, r_high)
    copy!(r_low, r_high)
    copy!(r_old_low, r_high)
    copy!(x_low, x_high)

    outer_iters = 0
    inner_iters = 0

    while inner_iters <= maxiters && outer_iters <= maxiters
        while res_low >= delta^2*res_high && inner_iters <= maxiters
            mul!(Ap_low, A_low, p_low)
            α = res_low / real(dot(p_low, Ap_low))
            axpy!(α, p_low, x_low)
            axpy!(-α, Ap_low, r_low)
            res_new = real(dot(r_low, r_low))

            β = (res_new - real(dot(r_low, r_old_low))) / res_low
            # β = res_new / res_low
            axpby!(1, r_low, β, p_low)
            res_low = res_new
            copy!(r_old_low, r_low)
            inner_iters += 1
            @level4 "|  Mixed CG: residual inner $(inner_iters) = $(sqrt(res_low))"
        end

        axpy!(1, x_low, x_high)
        mul!(Ap_high, A_high, x_high)
        copy!(r_high, b_high)
        axpy!(-1, Ap_high, r_high)
        res_high = real(dot(r_high, r_high))
        outer_iters += 1

        @level4 "|  Mixed CG: residual outer $(outer_iters) inner $(inner_iters) = $(sqrt(res_high))"

        if sqrt(res_high) < rel_tol
            @level3 "|  Mixed CG: converged at outer $(outer_iters) inner $(inner_iters) with res = $(sqrt(res_high))"
            print_solverdata(datafile, inner_iters, sqrt(res_high))
            return inner_iters, sqrt(res_high)
        end

        res_low = res_high
        empty!(x_low)
        copy!(r_low, r_high)
        copy!(r_old_low, r_high)
        axpby!(1, r_low, res_high/res_high_old, p_low)
        res_high_old = res_high
    end

    print_solverdata(datafile, maxiters, sqrt(res_high))
    throw(AssertionError("Mixed CG did not converge in $maxiters inner iterations"))
    return maxiters, sqrt(res_high)
end

function mscg_mixed!(
    x_high::NTuple{M}, shifts, A_high, A_low, b_high, Ap_high, r_high,
    x_low::NTuple{M}, p_low::NTuple{L}, r_low, Ap_low, r_old_low;
    tol=1e-7, maxiters=1000, datafile="", delta=0.1
) where {M,L} # multishift solver
    rel_tol = tol * sqrt(real(dot(b_high, b_high)))
    @assert all(x -> x>0, shifts) "Mixed precision multishift CG not supported for shifts < 0"
    N = length(shifts) + 1
    @assert L ≥ M ≥ N
    α = one(ComplexF64)
    β = zero(ComplexF64)
    α′ = @SVector ones(ComplexF64, N - 1)
    ρ′ = @SVector ones(ComplexF64, N - 1)
    γ′ = @SVector ones(ComplexF64, N - 1)
    β′ = @SVector zeros(ComplexF64, N - 1)

    mul!(Ap_high, A_high, x_high[1])
    copy!(r_high, b_high)
    axpy!(-1, Ap_high, r_high)

    for i in 1:N
        copy!(p_low[i], r_high)
        copy!(x_low[i], x_high[1])
    end

    res_high = dot(r_high, r_high)
    res_low′ = @SVector fill(res_high, N - 1)
    res_high_old = res_high
    res_low = res_high
    res_max_low = res_high

    if sqrt(abs(res_high)) < rel_tol
        @level3 "|  Mixed MSCG: converged at iter 0 with res = $(sqrt(abs(res_high)))"
        print_solverdata(datafile, 0, sqrt(abs(res_high)))
        return 0, sqrt(abs(res_high))
    end

    @level4 "|  Mixed MSCG: residual 0 = $(sqrt(abs(res_high)))"

    copy!(r_low, r_high)
    copy!(r_old_low, r_high)

    outer_iters = 0
    inner_iters = 0

    while inner_iters <= maxiters && outer_iters <= maxiters
        while abs(res_max_low) >= delta^2*abs(res_high) && inner_iters <= maxiters
            mul!(Ap_low, A_low, p_low[1])
            α_new = res_low / dot(p_low[1], Ap_low)
            ω = (α_new * β) / α
            ρ′ = 1 ./ (1 .+ shifts * α_new .+ (1 .- ρ′) .* ω)
            α′ = α_new * ρ′
            axpy!(α_new, p_low[1], x_low[1])
            axpy!(-α_new, Ap_low, r_low)
            res_new = dot(r_low, r_low)
            α = α_new
            β = (res_new - dot(r_low, r_old_low)) / res_low
            res_max_low = abs(res_new)

            for i in 1:N-1
                sqrt(abs(res_low′[i])) < rel_tol && continue
                axpy!(α′[i], p_low[i+1], x_low[i+1])
                @reset β′[i] = ρ′[i]^2 * β
                resᵢ = γ′[i] * res_new
                @reset res_low′[i] = resᵢ
                @reset γ′[i] = ρ′[i] * γ′[i]
                res_max_low = abs(resᵢ) > res_max_low ? abs(resᵢ) : res_max_low
            end

            axpby!(1, r_low, β, p_low[1])

            for i in 1:N-1
                sqrt(abs(res_low′[i])) < rel_tol && continue
                axpby!(γ′[i], r_low, β′[i], p_low[i+1])
            end

            copy!(r_old_low, r_low)
            inner_iters += 1
            res_low = res_new
            @level4 "|  Mixed MSCG: residual inner $(inner_iters) = $(sqrt(res_max_low))"
        end

        for i in 1:N
            axpy!(1, x_low[i], x_high[i])
        end

        mul!(Ap_high, A_high, x_high[1])
        copy!(r_high, b_high)
        axpy!(-1, Ap_high, r_high)
        res_high = real(dot(r_high, r_high))
        outer_iters += 1

        @level4 "|  Mixed MSCG: residual outer $(outer_iters) inner $(inner_iters) = $(sqrt(abs(res_high)))"

        if sqrt(abs(res_high)) < rel_tol
            @level3 "|  Mixed MSCG: converged at outer $(outer_iters) inner $(inner_iters) with res = $(sqrt(abs(res_high)))"
            print_solverdata(datafile, inner_iters, sqrt(abs(res_high)))
            return inner_iters, sqrt(abs(res_high))
        end

        res_low = res_high
        for i in 1:N
            empty!(x_low[i])
        end
        copy!(r_low, r_high)
        copy!(r_old_low, r_high)
        axpby!(1, r_low, res_high/res_high_old, p_low[1])
        res_high_old = res_high
    end

    print_solverdata(datafile, maxiters, sqrt(abs(res_high)))
    throw(AssertionError("Mixed MSCG did not converge in $maxiters iterations"))
    return maxiters, sqrt(abs(res_high))
end
