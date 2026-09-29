# https://arxiv.org/pdf/1908.10950
function solve_via_secant!(
    U::Gaugefield{B,T}, hmc, bias, c_final; Δτ=0.1, tol=1e-9, maxiters=100
) where {B,T}
    U0 = hmc.U0
    P = hmc.P
    P0 = hmc.P0
    force = hmc.force
    temp_force = hmc.force2
    F = (hmc.fieldstrength, hmc.staples)
    copy!(P0, P)
    copy!(U0, U)

    function f(U, U0, force, λ)
        add!(P, P0, force, Δτ/2*λ) # P -> P_0 - Δτ/2 * ∇S - Δτ/2 * λ * ∇c
        parallelfor(allindices(U, P), B, Val(false), (), (U,), (U, U0, P)) do μsite, (U, U0, P)
            U[μsite] = proj_onto_SU3(cmatmul_oo(exp_iQ(-im * Δτ * P[μsite]), ComplexF64.(U0[μsite])))
        end
        return calc_cv(U, bias)[1] - c_final
    end

    calc_cv(U0, bias)[1]
    calc_cv_deriv_bare!(force, bias, F, U0, temp_force, bias.smearing, 1)

    λ_old = 0.0
    fλ_old = f(U, U0, force, λ_old)
    rel_tol = tol

    if abs(fλ_old) < rel_tol
        # @show 0, abs(fλ_old), λ_old
        calc_cv_deriv_bare!(temp_force, bias, F, U, force, bias.smearing, 1) # for next step in RATTLE
        return λ_old
    end

    λ = 0.01
    fλ = f(U, U0, force, λ)
    res = abs(fλ)

    if res < rel_tol
        # @show 1, res, λ
        calc_cv_deriv_bare!(temp_force, bias, F, U, force, bias.smearing, 1) # for next step in RATTLE
        return λ
    end

    iters = 1

    for iter in 2:maxiters
        λ_new = λ - fλ*(λ - λ_old) / (fλ - fλ_old)
        fλ_new = f(U, U0, force, λ_new)
        res = abs(fλ_new)
        if res < rel_tol
            # @show iter, res, λ_new
            λ = λ_new
            fλ = fλ_new
            break
        end
        iters += 1

        λ_old = λ
        λ = λ_new
        fλ_old = fλ
        fλ = fλ_new
    end

    if iters == maxiters
        @warn "secant method did not converge in $maxiters iterations, continuing with res $res anyway"
    end

    calc_cv_deriv_bare!(temp_force, bias, F, U, force, bias.smearing, 1) # for next step in RATTLE
    return λ
end

function enforce_hidden_constraint!(hmc, U, bias)
    P = hmc.P
    force = hmc.force
    temp_force = hmc.force2
    F = (hmc.fieldstrength, hmc.staples)
    calc_cv(U, bias)
    calc_cv_deriv_bare!(force, bias, F, U, temp_force, bias.smearing, 1)
    fac = real(-6dot(force, P)) / real(-6dot(force, force))
    add!(P, force, -fac)
    cons = real(6dot(hmc.force, hmc.P))
    @assert abs(cons) < 1e-4 "hidden constraint was $cons"
    return nothing
end
