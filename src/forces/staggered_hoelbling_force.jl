function calc_dSfdU!(
    dU, fermion_action::StaggeredHoelblingFermionAction{false,2}, U, ϕ::StaggeredSpinorfield
)
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    term = get_mass_term(D)
    bc = D.boundary_condition

    clear!(X) # initial guess is zero
    solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp
    LinearAlgebra.mul!(Y, D, X)

    # X̃ = fermion_action.X̃
    # Ỹ = fermion_action.Ỹ
    # apply_ημν!(X̃, X)
    # apply_ημν!(Ỹ, Ỹ)
    add_staggered_hoelbling_derivative!(dU, U, X, Y, bc, term)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::StaggeredHoelblingFermionAction{true,1}, U, ϕ::StaggeredSpinorfield
)
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    rhmc = fermion_action.rhmc_info_md
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    term = get_mass_term(D)
    bc = D.boundary_condition
    Xs = fermion_action.rhmc_temps1
    Ys = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    solve_dirac_multishift!(Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        X = Xs[i+1]
        Y = Ys[i+1]
        LinearAlgebra.mul!(Y, D, X)
        add_staggered_hoelbling_derivative!(dU, U, X, Y, bc, term; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_hoelbling_derivative!(
    dU::Colorfield{CPU,T,M}, U::Gaugefield{CPU,T,M}, X::TF, Y::TF, bc, term; coeff=1
) where {T,M,TF<:StaggeredSpinorfield{CPU,T,M}}
    check_dims(dU, U, X, Y)
    fac1 = T(-0.5coeff)
    fac2 = T(coeff)
    _μ, _ν, _ρ, _σ = term

    @batch for site in eachindex(dU)
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac1)
        add_hoelbling_derivative_kernel!(dU, _μ, _ν, U, X, Y, site, bc, fac2)
        add_hoelbling_derivative_kernel!(dU, _ρ, _σ, U, X, Y, site, bc, fac2)
    end

    update_halo!(dU)
    return nothing
end

function add_hoelbling_derivative_kernel!(
    dU, ::Val{μ}, ::Val{ν}, U, X, Y, site, bc, fac::T
) where {μ,ν,T}
    B = Y∇MμνX(X, Y, U, Val(μ), Val(ν), site, bc, T)
    C = Y∇MμνX(Y, X, U, Val(μ), Val(ν), site, bc, T)
    dU[μ, site] += fac * traceless_antihermitian(cmatmul_oo(U[μ, site], B + C))

    B = Y∇MμνX(X, Y, U, Val(ν), Val(μ), site, bc, T)
    C = Y∇MμνX(Y, X, U, Val(ν), Val(μ), site, bc, T)
    dU[ν, site] += fac * traceless_antihermitian(cmatmul_oo(U[ν, site], B + C))
    return nothing
end

function Y∇MμνX(X, Y, U, ::Val{μ}, ::Val{ν}, site, bc, ::Type{T}) where {μ,ν,T}
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁻ = move(site, ν, -1, Nν)
    siteν⁺ = move(site, ν, 1, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)

    # Start
    η1 = im * T(1/8 * staggered_ημν(Val(μ), Val(ν), site))
    η2 = im * T(1/8 * staggered_ημν(Val(μ), Val(ν), siteν⁺))
    η3 = im * T(1/8 * staggered_ημν(Val(μ), Val(ν), siteν⁻))
    Y1 = η1 * Y[site]
    Y2 = η2 * apply_bc(Y[siteν⁺], bc, site, Val(1), Nν, Val(ν))
    Y3 = η3 * apply_bc(Y[siteν⁻], bc, site, Val(-1), Nν, Val(ν))
    
    # Stop
    X1 = apply_bc(
        apply_bc(X[siteμ⁺ν⁺], bc, site, Val(1), Nμ, Val(μ)),
        bc, site, Val(1), Nν, Val(ν)
    )
    X2 = apply_bc(
        apply_bc(X[siteμ⁺ν⁻], bc, site, Val(1), Nμ, Val(μ)),
        bc, site, Val(-1), Nν, Val(ν)
    )
    X3 = apply_bc(X[siteμ⁺], bc, site, Val(1), Nμ, Val(μ))

    out =
        cmatmul_oo(U[ν, siteμ⁺], ckron(X1, Y1)) + # site -> siteμ+ -> siteμ+ν+
        cmatmul_do(U[ν, siteμ⁺ν⁻], ckron(X2, Y1)) + # site -> siteμ+ -> siteμ+ν-
        cmatmul_od(ckron(X3, Y2), U[ν, site]) + # siteν+ -> site -> siteμ+
        cmatmul_oo(ckron(X3, Y3), U[ν, siteν⁻]) # siteν- -> site -> siteμ+

    return out
end
