module CG
using LinearAlgebra
using ..Output

export bicg!, bicg_stab!, cg!

function cg!(x, b, A, Ap, r, p; tol=1e-12, maxiters=1000)
    mul!(Ap, A, x)
    copy!(r, b)
    axpy!(-1, Ap, r)
    copy!(p, r)
    res = real(dot(r, r))
    if res < tol
        @level2 "|  CG: converged at iter 0 with res = $res_new"
        return nothing
    end
    @level3 "|  CG: residual 0 = $res"

    for iter in 1:maxiters
        mul!(Ap, A, p)
        α = res / real(dot(p, Ap))
        axpy!(α, p, x)
        axpy!(-α, Ap, r)
        res_new = real(dot(r, r))
        @level3 "|  CG: residual $(iter) = $res_new"
        if res_new < tol
            @level2 "|  CG: converged at iter $(iter) with res = $res_new"
            return nothing
        end
        β = res_new / res
        axpby!(1, r, β, p)
        res = res_new
    end
    @level1 "|  CG: did not converge in $maxiters iterations"
    # throw(AssertionError("CG did not converge in $maxiters iterations"))
    return nothing
end

function shifted_cg! end

function bicg!(x, b, A, Ap, r, p, Ap′, r′, p′; tol=1e-14, maxiters=1000)
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
    if res < tol
        @level2 "|  BiCG: converged at iter 0 with res = $res_new"
        return nothing
    end
    @level3 "|  BiCG: residual 0 = $res"

    for iter in 1:maxiters
        mul!(Ap, A, p)
        mul!(Ap′, adjoint(A), p′)
        α = ρ / dot(p′, Ap)
        axpy!(α, p, x)
        axpy!(-α, Ap, r)
        axpy!(-α, Ap′, r′)
        ρ_new = dot(r′, r)
        res = abs(dot(r, r))
        @level3 "|  BiCG: residual $(iter) = $res"
        if res < tol
            @level2 "|  BiCG: converged at iter $(iter) with res = $res"
            return nothing
        end
        β = ρ_new / ρ
        axpby!(1, r, β, p)
        axpby!(1, r′, β, p′)
        ρ = ρ_new
    end
    @level1 "|  BiCG: did not converge in $maxiters iterations"
    # throw(AssertionError("BiCG did not converge in $maxiters iterations"))
    return nothing
end

function bicg_stab!(x, b, A, v, r, p, r₀, t; tol=1e-14, maxiters=1000)
    mul!(v, A, x)
    copy!(r, b)
    axpy!(-1, v, r)
    copy!(r₀, r)
    copy!(p, r)
    ρ = dot(r₀, r)
    res = abs(ρ)
    @level1 "|  BiCGStab: residual 0 = $res"
    res < tol && return nothing
    @assert isfinite(res) && isfinite(ρ) "BiCG: NaN or Inf encountered"

    for iter in 1:maxiters
        mul!(v, A, p)
        α = ρ / dot(r₀, v)
        axpy!(α, p, x)
        axpy!(-α, v, r)
        res = abs(dot(r, r))
        @level3 "|  BiCGStab: residual $(iter).5 = $res"
        if res < tol
            @level2 "|  BiCGStab: converged at iter $(iter).5 with res = $res"
            return nothing
        end
        @assert isfinite(res) && isfinite(α) "BiCG: NaN or Inf encountered, res = $res, α = $α"
        mul!(t, A, r)
        ω = dot(t, r) / dot(t, t)
        axpy!(ω, r, x)
        axpy!(-ω, t, r)
        res = abs(dot(r, r))
        @level3 "|  BiCGStab: residual $(iter) = $res"
        if res < tol
            @level2 "|  BiCGStab: converged at iter $(iter) with res = $res"
            return nothing
        end
        @assert isfinite(res) && isfinite(ω) "BiCG: NaN or Inf encountered, res = $res, ω = $ω"
        ρ_new = dot(r₀, r)
        β = (ρ_new / ρ) * (α / ω)
        axpy!(-ω, v, p)
        axpby!(1, r, β, p)
        ρ = ρ_new
    end
    @level1 "|  BiCGStab: did not converge in $maxiters iterations"
    return nothing
end

end # module CG
