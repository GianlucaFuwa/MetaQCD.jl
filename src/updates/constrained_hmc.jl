# https://arxiv.org/pdf/1908.10950
function solve_via_secant!(
    U::Gaugefield{B,T}, hmc, bias, c_final; Δτ=0.1, tol=1e-9, maxiters=100, λ
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
        return calc_cv(U, bias, 1) - c_final
    end

    calc_cv(U0, bias, 1)
    calc_cv_deriv_bare!(force, bias, F, U0, temp_force, bias.smearing, 1)
    # @show log(real(-6dot(force, force)))

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

function enforce_momentum_constraint!(P, U, F, force, temp_force, stout)
    calc_smearedU!(stout, U)
    top_charge_deriv!(force, F, U, Clover())
    stout_backprop!(force, temp_force, stout)
    fac = real(dot(force, P)) / real(dot(force, force))
    axpy!(-fac, force, P)
    return nothing
end

function newton_raphsons_Q!(
    U, Utmp::Gaugefield{B,T}, fieldstrength, force, c_final; tol=1e-3, maxiters=100
) where {B,T}
    copy!(Utmp, U)
    rel_tol = c_final * tol
    λ = 0.0
    iters = 0
    δc = top_charge(Clover(), U) - c_final
    res = abs(δc)
    res < rel_tol && return nothing

    for iter in 1:maxiters
        top_charge_deriv!(force, fieldstrength, U, Clover())
        λ += δc * 0.01#-c_current#/norm(force, Val(2))

        parallelfor(allindices(U, force), B, Val(false), (), (U,), (U, force)) do μsite, (U, force)
            U[μsite] = proj_onto_SU3(cmatmul_oo(exp_iQ(-im * T(λ) * force[μsite]), Utmp[μsite]))
        end

        c_current = top_charge(Clover(), U)
        δc = c_current - c_final
        res = abs(δc)
        # @show iter, res, c_current
        res < rel_tol && break
    end

    if iters == maxiters
        @warn "newton-raphsons did not converge in $maxiters iterations, continuing with res $res anyway"
    end

    return nothing
end

# function solve_via_secant!(
#     U::Gaugefield{B,T}, U0, stout, force, temp_force, F, c_final; h=0.01, tol=1e-6, maxiters=100
# ) where {B,T}
#     Usmeared = stout.Usmeared_multi[end]
#     P0 = Colorfield(U)
#     P = Colorfield(U)
#     calc_P0!(P, U, F, force, temp_force, stout)
#     function f(U, U0, force, l)
#         add!(P, P0, force, h/2*(1+l))
#         parallelfor(allindices(U, P), B, Val(false), (), (U,), (U, U0, P)) do μsite, (U, U0, P)
#             U[μsite] = proj_onto_SU3(cmatmul_oo(exp_iQ(-im * h * P[μsite]), U0[μsite]))
#         end
#         calc_smearedU!(stout, U)
#         return top_charge(Clover(), Usmeared) - c_final
#     end
#
#     copy!(U0, U)
#     # top_charge_deriv!(force, F, U, Clover())
#     # stout_backprop!(force, temp_force, stout)
#     λ_old = 0.0
#     λ = 0.1
#     fλ_old = f(U, U0, force, λ_old)
#     fλ = f(U, U0, force, λ)
#
#     rel_tol = c_final * tol
#     iters = 0
#     res = abs(fλ)
#
#     for _ in 1:maxiters
#         λ_new = λ - fλ*(λ - λ_old) / (fλ - fλ_old)
#         fλ_new = f(U, U0, force, λ_new)
#         res = abs(fλ_new)
#         # @show iters, res, fλ_new
#         res < rel_tol && break
#         iters += 1
#
#         λ_old = λ
#         λ = λ_new
#         fλ_old = fλ
#         fλ = fλ_new
#     end
#
#     if iters == maxiters
#         @warn "newton-raphsons did not converge in $maxiters iterations, continuing with res $res anyway"
#     end
#
#     return λ
# end
#
# function get_lambda0_S(U, P, force, Δτ, c_final)
#     t2 = Δτ^2/2
#     denom = real(dot(force, U))
#     lambda0 = 2denom/Δτ^2 * (
#         -c_final + real(tr(U)) + Δτ*real(dot(P, U)) - t2*denom + t2*real(dot(P, U))
#     )
#     return real(lambda0)
# end
#
# function newton_raphsons_S!(
#     U::Gaugefield{B,T}, Utmp, force, c_final; tol=1e-3, maxiters=100
# ) where {B,T}
#     copy!(Utmp, U)
#     λ = 0.0
#     rel_tol = c_final * tol
#     iters = 0
#     res = abs(calc_gauge_action(U) / 18length(U) - c_final)
#
#     for _ in 1:maxiters
#         calcZ!(force, U, 1)
#         c_current = calc_gauge_action(U) / 18length(U)
#         δc = c_current - c_final
#         res = abs(δc)
#         # @show iters, res, c_current
#         res < rel_tol && break
#         λ += δc * 0.01#= /norm(force, Val(2))^2 =#
#
#         parallelfor(allindices(U, force), B, Val(false), (), (U,), (U, force)) do μsite, (U, force)
#             U[μsite] = proj_onto_SU3(cmatmul_oo(exp_iQ(-im * T(λ) * force[μsite]), Utmp[μsite]))
#         end
#
#         iters += 1
#     end
#
#     if iters == maxiters
#         @warn "newton-raphsons did not converge in $maxiters iterations, continuing with res $res anyway"
#     end
#
#     return λ
# end
#
