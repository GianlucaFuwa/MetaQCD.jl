function calc_dSfdU!( # Force for unrooted Staggered-Hoelbling Action (Nf=2)
    dU,
    fermion_action::FermionAction{false,2,TD},
    U,
    ϕ::StaggeredSpinorfield,
) where {TD<:StaggeredHoelblingDiracOperator}
    clear!(dU)
    X, Y, temp1, temp2 = fermion_action.temps[1:4]
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    term = get_mass_term(D)
    bc = D.boundary_condition
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    clear!(X) # initial guess is zero
    solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2; tol, maxiters, datafile)

    LinearAlgebra.mul!(Y, D, X)
    add_staggered_hoelbling_derivative!(dU, U, X, Y, bc, term)
    return nothing
end

function calc_dSfdU!( # Force for single flavor Staggered-Hoelbling Action (Nf=2)
    dU,
    fermion_action::FermionAction{true,1,TD},
    U,
    ϕ::StaggeredSpinorfield,
) where {TD<:StaggeredHoelblingDiracOperator}
    clear!(dU)
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    term = get_mass_term(D)
    bc = D.boundary_condition
    temp1, temp2 = fermion_action.temps[1:2]
    Xs = fermion_action.temps[3:n+3]
    Ys = fermion_action.temps[n+4:2n+4]
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    solve_dirac_multishift!(
        Xs, shifts, DdagD, ϕ, temp1, temp2, Ys; tol, maxiters, datafile
    )

    for i in 1:n
        X = Xs[i+1]
        Y = Ys[i+1]
        LinearAlgebra.mul!(Y, D, X)
        add_staggered_hoelbling_derivative!(dU, U, X, Y, bc, term; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_hoelbling_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T,M}, X::TF, Y::TF, bc, term; coeff=1
) where {B,T,M,TF<:StaggeredSpinorfield{B,T,M}}
    fac1 = T(-0.5coeff)
    fac2 = T(coeff)
    _μ, _ν, _ρ, _σ = term
    itr = eachindex(dU, U, X, Y)

    parallelfor(itr, B, Val(M), (U, X, Y), (dU,), (dU, U, X, Y)) do site, (dU, U, X, Y)
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac1)
        add_hoelbling_derivative_kernel!(dU, _μ, _ν, U, X, Y, site, bc, fac2)
        add_hoelbling_derivative_kernel!(dU, _ρ, _σ, U, X, Y, site, bc, fac2)
    end

    return nothing
end

function add_hoelbling_derivative_kernel!(
    dU, ::Val{μ}, ::Val{ν}, U, X, Y, site, bc, fac::T
) where {μ,ν,T}
    B = Y∇MμνX(X, Y, U, Val(μ), Val(ν), site, bc, T)
    C = Y∇MμνX(Y, X, U, Val(μ), Val(ν), site, bc, T)
    @inbounds dU[μ, site] += fac * traceless_antihermitian(cmatmul_oo(U[μ, site], B + C))

    B = Y∇MμνX(X, Y, U, Val(ν), Val(μ), site, bc, T)
    C = Y∇MμνX(Y, X, U, Val(ν), Val(μ), site, bc, T)
    @inbounds dU[ν, site] += fac * traceless_antihermitian(cmatmul_oo(U[ν, site], B + C))
    return nothing
end

function Y∇MμνX(X, Y, U, ::Val{μ}, ::Val{ν}, site, bc, ::Type{T}) where {μ,ν,T}
    NT = size(U, 4)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁻ = move(site, ν, -1, Nν)
    siteν⁺ = move(site, ν, 1, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)

    @inbounds begin
        # Start
        η1 = im * T(1 / 8 * staggered_ημν(Val(μ), Val(ν), site, Val(true)))
        η2 = im * T(1 / 8 * staggered_ημν(Val(μ), Val(ν), siteν⁺, Val(true)))
        η3 = im * T(1 / 8 * staggered_ημν(Val(μ), Val(ν), siteν⁻, Val(true)))
        Y1 = η1 * Y[site]
        Y2 = η2 * apply_bc(Y[siteν⁺], bc, site, Val(1), NT, Val(ν))
        Y3 = η3 * apply_bc(Y[siteν⁻], bc, site, Val(-1), NT, Val(ν))

        # Stop
        X1 = apply_bc(
            apply_bc(X[siteμ⁺ν⁺], bc, site, Val(1), NT, Val(μ)), bc, site, Val(1), NT, Val(ν)
        )
        X2 = apply_bc(
            apply_bc(X[siteμ⁺ν⁻], bc, site, Val(1), NT, Val(μ)), bc, site, Val(-1), NT, Val(ν)
        )
        X3 = apply_bc(X[siteμ⁺], bc, site, Val(1), NT, Val(μ))

        out =
            cmatmul_oo(U[ν, siteμ⁺], ckron(X1, Y1)) + # site -> siteμ+ -> siteμ+ν+
            cmatmul_do(U[ν, siteμ⁺ν⁻], ckron(X2, Y1)) + # site -> siteμ+ -> siteμ+ν-
            cmatmul_od(ckron(X3, Y2), U[ν, site]) + # siteν+ -> site -> siteμ+
            cmatmul_oo(ckron(X3, Y3), U[ν, siteν⁻]) # siteν- -> site -> siteμ+
    end

    return out
end
