module CG
using LinearAlgebra
using ..Output

export cg!, set_cg_max_iterations!, set_cg_tolerance!

const TOLERANCE = Base.RefValue{Float64}(1e-10)
const MAX_ITERATIONS = Base.RefValue{Int64}(100)

function set_cg_tolerance!(val)
    TOLERANCE[] = val
    return nothing
end

function set_cg_max_iterations!(val)
    MAX_ITERATIONS[] = val
    return nothing
end

function cg!(x, b, A, Ap, r, p; tol=TOLERANCE[], maxiter=1000)
    mul!(Ap, A, x)
    copy!(r, b)
    axpy!(-1, Ap, r)
    copy!(p, r)
    res = real(dot(r, r))
    @level1 "|  CG: initial residual = $res"
    res < tol && return nothing

    for iter in 1:maxiter
        mul!(Ap, A, p)
        α = res / real(dot(p, Ap))
        axpy!(α, p, x)
        axpy!(-α, Ap, r)
        res_new = real(dot(r, r))
        @level1 "|  CG: iteration $iter, residual = $res_new"
        res_new < tol && return nothing
        β = res_new / res
        axpby!(1, r, β, p)
        res = res_new
    end
    @level1 "|  CG: did not converge in $maxiter iterations"
    # throw(AssertionError("CG did not converge in $maxiter iterations"))
    return nothing
end

function shifted_cg! end

function bicg! end

function bicg_stab! end

end # module CG
