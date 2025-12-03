function cg!(x, A, b, Ap, r, p; tol=1e-7, maxiters=1000, datafile="")
    rel_tol = tol * sqrt(real(dot(b, b)))
    mul!(Ap, A, x)
    copy!(r, b)
    axpy!(-1, Ap, r)
    copy!(p, r)
    res = real(dot(r, r))

    if sqrt(res) < rel_tol
        @level3 "|  CG: converged at iter 0 with res = $(sqrt(res))"
        print_solverdata(datafile, 0, sqrt(res))
        return 0, sqrt(res)
    end

    @level4 "|  CG: residual 0 = $(sqrt(res))"

    for iter in 1:maxiters
        mul!(Ap, A, p)
        α = res / real(dot(p, Ap))
        axpy!(α, p, x)
        axpy!(-α, Ap, r)
        res_new = real(dot(r, r))
        @level4 "|  CG: residual $(iter) = $(sqrt(res_new))"

        if sqrt(res_new) < rel_tol
            @level3 "|  CG: converged at iter $(iter) with res = $(sqrt(res_new))"
            print_solverdata(datafile, iter, sqrt(res_new))
            return iter, sqrt(res_new)
        end

        β = res_new / res
        axpby!(1, r, β, p)
        res = res_new
    end

    print_solverdata(datafile, maxiters, sqrt(res))
    throw(AssertionError("CG did not converge in $maxiters iterations"))
    return maxiters, sqrt(res)
end

function cgnr!(x, A, A_dagg, b, Ap, r, g, p; tol=1e-12, maxiters=1000, datafile="")
    rel_tol = tol * sqrt(real(dot(b, b)))
    mul!(Ap, A, x)
    copy!(r, b)
    axpy!(-1, Ap, r)

    res = real(dot(r, r))
    res_new = res
    gres = res
    gres_new = gres

    for iter in 1:maxiters+1
        @level4 "|  CGNR: residual $(iter-1) = $(sqrt(res_new))"

        if sqrt(res_new) < rel_tol
            @level3 "|  CGNR: converged at iter $(iter-1) with res = $(sqrt(res_new))"
            print_solverdata(datafile, iter-1, sqrt(res_new))
            return iter, sqrt(res_new)
        end

        mul!(g, A_dagg, r)
        gres_new = real(dot(g, g))

        if iter == 1
            copy!(p, g)
        else
            β = gres_new / gres
            axpby!(1, g, β, p)
        end

        mul!(Ap, A, p)

        α = gres_new / real(dot(Ap, Ap))
        axpy!(α, p, x)
        axpy!(-α, Ap, r)

        res = res_new
        gres = gres_new

        res_new = real(dot(r, r))
    end

    print_solverdata(datafile, maxiters, sqrt(res_new))
    throw(AssertionError("CGNR did not converge in $maxiters iterations"))
    return maxiters, sqrt(res_new)
end

function mscg!(
    x::NTuple{M,V}, shifts, A, b::V, Ap::V, r::V, p::NTuple{L,V};
    tol=1e-7, maxiters=1000, datafile=""
) where {M,L,V} # multishift solver
    rel_tol = tol * sqrt(real(dot(b, b)))
    N = length(shifts) + 1
    @assert L ≥ M ≥ N
    α = one(ComplexF64)
    β = zero(ComplexF64)
    α′ = @SVector ones(ComplexF64, N - 1)
    ρ′ = @SVector ones(ComplexF64, N - 1)
    γ′ = @SVector ones(ComplexF64, N - 1)
    β′ = @SVector zeros(ComplexF64, N - 1)

    mul!(Ap, A, x[1])
    copy!(r, b)
    axpy!(-1, Ap, r)

    for i in 1:N
        copy!(p[i], r)
    end

    res = dot(r, r)
    res′ = @SVector fill(res, N - 1)

    if sqrt(abs(res)) < rel_tol
        @level3 "|  MultishiftCG: converged at iter 0 with res = $(sqrt(abs(res)))"
        print_solverdata(datafile, 0, sqrt(abs(res)))
        return 0, sqrt(abs(res))
    end

    @level4 "|  MultishiftCG: residual 0 = $(sqrt(abs(res)))"

    for iter in 1:maxiters
        mul!(Ap, A, p[1])
        α_new = res / dot(p[1], Ap)
        ω = (α_new * β) / α
        ρ′ = 1 ./ (1 .+ shifts * α_new .+ (1 .- ρ′) .* ω)
        α′ = α_new * ρ′
        axpy!(α_new, p[1], x[1])
        axpy!(-α_new, Ap, r)
        res_new = dot(r, r)
        α = α_new
        β = res_new / res
        res_max = abs(res_new)

        for i in 1:N-1
            sqrt(abs(res′[i])) < rel_tol && continue
            axpy!(α′[i], p[i+1], x[i+1])
            @reset β′[i] = ρ′[i]^2 * β
            resᵢ = γ′[i] * res_new
            @reset res′[i] = resᵢ
            @reset γ′[i] = ρ′[i] * γ′[i]
            res_max = abs(resᵢ) > res_max ? abs(resᵢ) : res_max
        end

        @level4 "|  MultishiftCG: max residual $(iter) = $(sqrt(res_max))"

        if sqrt(res_max) < rel_tol
            @level3 "|  MultishiftCG: converged at iter $(iter) with res = $(sqrt(res_max))"
            print_solverdata(datafile, iter, sqrt(abs(res_max)))
            return iter, sqrt(res_max)
        end

        axpby!(1, r, β, p[1])

        for i in 1:N-1
            sqrt(abs(res′[i])) < rel_tol && continue
            axpby!(γ′[i], r, β′[i], p[i+1])
        end

        res = res_new
    end

    print_solverdata(datafile, maxiters, sqrt(abs(res)))
    throw(AssertionError("MultishiftCG did not converge in $maxiters iterations"))
    return maxiters, sqrt(abs(res))
end

function bicg!(x, A, b, Ap, r, p, Ap′, r′, p′; tol=1e-7, maxiters=1000, datafile="")
    rel_tol = tol * sqrt(real(dot(b, b)))
    mul!(Ap, A, x)
    mul!(Ap′, adjoint(A), x)
    copy!(r, b)
    copy!(r′, b)
    axpy!(-1, Ap, r)
    axpy!(-1, Ap′, r′)
    copy!(p, r)
    copy!(p′, r′)
    ρ = dot(r′, r)
    res = abs(dot(r, r))

    if sqrt(res) < rel_tol
        @level3 "|  BiCG: converged at iter 0 with res = $(sqrt(res))"
        print_solverdata(datafile, 0, sqrt(res))
        return 0, sqrt(res)
    end

    @level4 "|  BiCG: residual 0 = $(sqrt(res))"

    for iter in 1:maxiters
        mul!(Ap, A, p)
        mul!(Ap′, adjoint(A), p′)
        α = ρ / dot(p′, Ap)
        axpy!(α, p, x)
        axpy!(-α, Ap, r)
        axpy!(-α, Ap′, r′)
        ρ_new = dot(r′, r)
        res = abs(dot(r, r))
        @level4 "|  BiCG: residual $(iter) = $(sqrt(res))"

        if res < rel_tol
            @level3 "|  BiCG: converged at iter $(iter) with res = $(sqrt(res))"
            print_solverdata(datafile, iter, sqrt(res))
            return iter, sqrt(res)
        end

        β = ρ_new / ρ
        axpby!(1, r, β, p)
        axpby!(1, r′, β, p′)
        ρ = ρ_new
    end

    print_solverdata(datafile, maxiters, sqrt(res))
    throw(AssertionError("BiCG did not converge in $maxiters iterations"))
    return maxiters, sqrt(res)
end

function bicg_stab!(x, A, b, v, r, p, r₀, t; tol=1e-7, maxiters=1000, datafile="")
    rel_tol = tol * sqrt(real(dot(b, b)))
    mul!(v, A, x)
    copy!(r, b)
    axpy!(-1, v, r)
    copy!(r₀, r)
    copy!(p, r)
    ρ = dot(r₀, r)
    res = abs(ρ)
    @level4 "|  BiCGStab: residual 0 = $(sqrt(res))"

    if res < rel_tol
        @level3 "|  BiCGStab: converged at iter 0 with res = $(sqrt(res))"
        print_solverdata(datafile, 0, sqrt(res))
        return 0, sqrt(res)
    end

    @assert isfinite(res) && isfinite(ρ) "BiCG: NaN or Inf encountered"

    for iter in 1:maxiters
        mul!(v, A, p)
        α = ρ / dot(r₀, v)
        axpy!(α, p, x)
        axpy!(-α, v, r)
        res = abs(dot(r, r))
        @level4 "|  BiCGStab: residual $(iter).5 = $(sqrt(res))"

        if res < rel_tol
            @level3 "|  BiCGStab: converged at iter $(iter).5 with res = $(sqrt(res))"
            print_solverdata(datafile, iter, sqrt(res))
            return iter, sqrt(res)
        end

        @assert isfinite(res) && isfinite(α) """
        BiCG: NaN or Inf encountered, res = $(sqrt(res)), α = $α
        """
        mul!(t, A, r)
        ω = dot(t, r) / dot(t, t)
        axpy!(ω, r, x)
        axpy!(-ω, t, r)
        res = abs(dot(r, r))
        @level4 "|  BiCGStab: residual $(iter) = $(sqrt(res))"

        if res < rel_tol
            @level3 "|  BiCGStab: converged at iter $(iter) with res = $(sqrt(res))"
            print_solverdata(datafile, iter, sqrt(res))
            return iter, sqrt(res)
        end

        @assert isfinite(res) && isfinite(ω) """
        BiCG: NaN or Inf encountered, res = $(sqrt(res)), ω = $ω
        """
        ρ_new = dot(r₀, r)
        β = (ρ_new / ρ) * (α / ω)
        axpy!(-ω, v, p)
        axpby!(1, r, β, p)
        ρ = ρ_new
    end

    print_solverdata(datafile, maxiters, sqrt(res))
    throw(AssertionError("BiCGStab did not converge in $maxiters iterations"))
    return maxiters, sqrt(res)
end
